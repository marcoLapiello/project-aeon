#include "server/http_server.hpp"

#include "server/json_write.hpp"
#include "server/openai_codec.hpp"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace aeon::server {

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxPayload = 8 * 1024 * 1024;  // the context bounds a valid prompt

void json_response(httplib::Response& res, int status, const std::string& body) {
    res.status = status;
    res.set_content(body, "application/json");
}

Usage usage_of(const session::ConversationResult& result) {
    Usage usage;
    usage.prompt_tokens = result.prompt_tokens;
    usage.completion_tokens = result.completion_tokens;
    usage.total_tokens = result.prompt_tokens + result.completion_tokens;
    return usage;
}

CodecError error_of(session::ErrorKind kind, const std::string& message) {
    switch (kind) {
        case session::ErrorKind::InvalidRequest:
            return CodecError{400, "invalid_request_error", message};
        case session::ErrorKind::ContextOverflow:
            return CodecError{400, "context_length_exceeded", message};
        case session::ErrorKind::Internal:
            break;
    }
    return CodecError{500, "internal_error", message};
}

}  // namespace

struct HttpServer::Impl {
    ConversationService& service;
    HttpOptions options;
    httplib::Server server;
    std::thread thread;
    int bound_port{0};

    Impl(ConversationService& service_ref, HttpOptions opts)
        : service(service_ref), options(std::move(opts)) {}

    std::string model_id() const { return service.snapshot().info.model_id; }

    void setup() {
        server.set_payload_max_length(kMaxPayload);
        server.set_read_timeout(std::chrono::seconds(5));
        // A streamed generation may pause between chunks; the queued keep-alive
        // keeps a short write timeout from firing, but give it headroom anyway.
        server.set_write_timeout(std::chrono::seconds(300));

        const size_t pool = options.max_queue + 8;
        server.new_task_queue = [pool] { return new httplib::ThreadPool(pool); };

        server.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                        std::exception_ptr) {
            json_response(res, 500,
                         error_json(CodecError{500, "internal_error", "internal server error"}));
        });
        server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
            if (!res.body.empty()) return;  // our own JSON error is already present
            const int status = res.status < 400 ? 500 : res.status;
            std::string code = "invalid_request_error";
            if (status == 404) code = "not_found";
            else if (status == 405) code = "method_not_allowed";
            else if (status == 413) code = "payload_too_large";
            if (status >= 500) code = "internal_error";
            json_response(res, status, error_json(CodecError{status, code, "request failed"}));
        });

        server.Post("/v1/chat/completions",
                    [this](const httplib::Request& req, httplib::Response& res) {
                        handle_chat(req, res);
                    });
        server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            const bool ready = service.state() == ConversationService::State::Ready;
            json_response(res, ready ? 200 : 503,
                          ready ? R"({"status":"ok"})" : R"({"status":"loading"})");
        });
        server.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            std::string body;
            JsonWriter writer(body);
            writer.begin_object()
                .key("object").string("list")
                .key("data").begin_array().begin_object()
                    .key("id").string(model_id())
                    .key("object").string("model")
                    .key("owned_by").string("aeon")
                .end_object().end_array()
                .end_object();
            json_response(res, 200, body);
        });
        server.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
            const ServerSnapshot snap = service.snapshot();
            std::string body;
            JsonWriter writer(body);
            writer.begin_object()
                .key("state").string(snap.state)
                .key("queue_depth").integer(static_cast<int64_t>(snap.queue_depth))
                .key("conversation").begin_object()
                    .key("model_id").string(snap.info.model_id)
                    .key("context_capacity").integer(snap.info.context_capacity)
                    .key("resident_tokens").integer(snap.info.resident_tokens)
                    .key("invariants_ok").boolean(snap.info.invariants_ok)
                .end_object();
            if (snap.live.has_value()) {
                writer.key("live").begin_object()
                    .key("id").integer(static_cast<int64_t>(snap.live->id))
                    .key("phase").string(snap.live->phase)
                    .key("tokens_so_far").integer(static_cast<int64_t>(snap.live->tokens_so_far))
                    .key("elapsed_ms").number(snap.live->elapsed_ms)
                .end_object();
            } else {
                writer.key("live").null_value();
            }
            if (snap.last.has_value()) {
                writer.key("last").begin_object()
                    .key("stop").string(text::stop_reason_name(snap.last->result.stop))
                    .key("prompt_tokens").integer(snap.last->result.prompt_tokens)
                    .key("completion_tokens").integer(snap.last->result.completion_tokens)
                    .key("reused_tokens").integer(snap.last->result.reused_tokens)
                    .key("prefilled_tokens").integer(snap.last->result.prefilled_tokens)
                    .key("reuse_verdict")
                        .string(session::PrefixRecord::verdict_name(snap.last->result.verdict))
                    .key("ttft_ms").number(snap.last->result.ttft_ms)
                    .key("decode_tokens_per_second")
                        .number(snap.last->result.decode_tokens_per_second)
                    .key("queue_wait_ms").number(snap.last->queue_wait_ms)
                    .key("total_ms").number(snap.last->total_ms)
                .end_object();
            } else {
                writer.key("last").null_value();
            }
            writer.end_object();
            json_response(res, 200, body);
        });
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        ParsedRequest parsed;
        try {
            parsed = parse_chat_request(req.body);
        } catch (const CodecError& error) {
            json_response(res, error.http_status, error_json(error));
            return;
        }

        std::shared_ptr<Job> job;
        try {
            job = service.submit(std::move(parsed.request));
        } catch (const QueueFull& error) {
            res.set_header("Retry-After", "5");
            json_response(res, 503, error_json(CodecError{503, "server_busy", error.what()}));
            return;
        } catch (const NotReady& error) {
            res.set_header("Retry-After", "5");
            json_response(res, 503, error_json(CodecError{503, "server_loading", error.what()}));
            return;
        }

        if (parsed.stream) {
            handle_stream(job, parsed, req, res);
        } else {
            handle_non_stream(job, parsed, req, res);
        }
    }

    void handle_stream(const std::shared_ptr<Job>& job, const ParsedRequest& parsed,
                       const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_header("Connection", "keep-alive");
        const std::string id = new_completion_id();
        const std::string model = model_id();
        const bool include_usage = parsed.include_usage;

        res.set_chunked_content_provider(
            "text/event-stream",
            [this, job, id, model, include_usage](size_t, httplib::DataSink& sink) -> bool {
                return stream_job(job, id, model, include_usage, sink);
            },
            [this, job](bool success) {
                if (!success) service.cancel(job);
            });
    }

    bool stream_job(const std::shared_ptr<Job>& job, const std::string& id,
                    const std::string& model, bool include_usage, httplib::DataSink& sink) {
        const auto write_raw = [&sink](const std::string& text) {
            return sink.write(text.data(), text.size());
        };
        const auto write_data = [&write_raw](const std::string& json) {
            return write_raw("data: " + json + "\n\n");
        };
        const auto abort = [this, &job]() {
            service.cancel(job);
            return false;
        };

        if (!write_data(chat_completion_chunk_role(id, model))) return abort();

        auto last_keepalive = Clock::now();
        for (;;) {
            JobEvent event;
            const bool got = job->stream->pop(event, std::chrono::milliseconds(100));
            if (!got) {
                if (!sink.is_writable()) return abort();
                if (job->stream->is_closed()) {
                    sink.done();
                    return true;
                }
                const size_t position = service.position(job);
                const auto now = Clock::now();
                if (position > 0 && now - last_keepalive >= std::chrono::seconds(1)) {
                    if (!write_raw(": queued position=" + std::to_string(position) + "\n\n")) {
                        return abort();
                    }
                    last_keepalive = now;
                }
                continue;
            }

            if (event.kind == JobEvent::Kind::Delta) {
                if (!write_data(chat_completion_chunk_delta(id, model, event.delta.channel,
                                                            event.delta.text))) {
                    return abort();
                }
            } else if (event.kind == JobEvent::Kind::Finished) {
                if (!write_data(chat_completion_chunk_finish(id, model, event.result.stop))) {
                    return abort();
                }
                if (include_usage &&
                    !write_data(chat_completion_chunk_usage(id, model, usage_of(event.result)))) {
                    return abort();
                }
                if (!write_raw("data: [DONE]\n\n")) return abort();
                sink.done();
                return true;
            } else {
                if (!write_data(error_json(error_of(event.error_kind, event.error_message)))) {
                    return abort();
                }
                sink.done();
                return true;
            }
        }
    }

    void handle_non_stream(const std::shared_ptr<Job>& job, const ParsedRequest& parsed,
                           const httplib::Request& req, httplib::Response& res) {
        std::string content;
        std::string reasoning;
        for (;;) {
            JobEvent event;
            const bool got = job->stream->pop(event, std::chrono::milliseconds(100));
            if (!got) {
                if (job->stream->is_closed()) break;
                if (req.is_connection_closed()) {
                    service.cancel(job);
                    res.status = 499;  // the client is gone; nothing is sent
                    return;
                }
                continue;
            }

            if (event.kind == JobEvent::Kind::Delta) {
                if (event.delta.channel == session::Channel::Reasoning) {
                    reasoning += event.delta.text;
                } else {
                    content += event.delta.text;
                }
            } else if (event.kind == JobEvent::Kind::Finished) {
                Timings timings;
                timings.ttft_ms = event.result.ttft_ms;
                timings.decode_tokens_per_second = event.result.decode_tokens_per_second;
                timings.reused_tokens = event.result.reused_tokens;
                timings.prefilled_tokens = event.result.prefilled_tokens;
                timings.reuse_verdict =
                    session::PrefixRecord::verdict_name(event.result.verdict);
                timings.seed = parsed.seed;
                json_response(res, 200,
                              chat_completion(new_completion_id(), model_id(), reasoning, content,
                                              event.result.stop, usage_of(event.result), timings));
                return;
            } else {
                const CodecError error = error_of(event.error_kind, event.error_message);
                json_response(res, error.http_status, error_json(error));
                return;
            }
        }
        json_response(res, 500,
                      error_json(CodecError{500, "internal_error",
                                            "the job stream closed without a result"}));
    }
};

HttpServer::HttpServer(ConversationService& service, HttpOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start() {
    impl_->setup();

    if (impl_->options.port == 0) {
        impl_->bound_port = impl_->server.bind_to_any_port(impl_->options.host);
        if (impl_->bound_port <= 0) {
            throw std::runtime_error("failed to bind " + impl_->options.host + ":0");
        }
    } else {
        if (!impl_->server.bind_to_port(impl_->options.host, impl_->options.port)) {
            throw std::runtime_error("failed to bind " + impl_->options.host + ":" +
                                     std::to_string(impl_->options.port));
        }
        impl_->bound_port = impl_->options.port;
    }

    impl_->thread = std::thread([this] { impl_->server.listen_after_bind(); });
    for (int i = 0; i < 500 && !impl_->server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int HttpServer::port() const { return impl_->bound_port; }

void HttpServer::stop() {
    if (!impl_) return;
    impl_->server.stop();
    if (impl_->thread.joinable()) impl_->thread.join();
}

}  // namespace aeon::server
