// -----------------------------------------------------------------------------
// Gate — the HTTP layer.
//
// Drives a real `httplib::Client` against the server on loopback with a fake
// engine and pins the requirements: streaming is not buffered (the latch case
// deadlocks a buffering server), a second request queues rather than interleaving,
// a client that leaves is cancelled, every bad input yields a defined JSON error,
// and `/health` and `/status` answer without waiting on the engine.
//
// The test is built by `aeon_add_host_test` and links no HIP runtime.
// -----------------------------------------------------------------------------

#include "server/http_server.hpp"

#include "infrastructure/json.hpp"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using aeon::core::JsonValue;
using aeon::server::ConversationService;
using aeon::server::HttpOptions;
using aeon::server::HttpServer;
using aeon::session::Channel;
using aeon::session::ConversationEngine;
using aeon::session::ConversationError;
using aeon::session::ConversationInfo;
using aeon::session::ConversationRequest;
using aeon::session::ConversationResult;
using aeon::session::DeltaSink;
using aeon::session::ErrorKind;
using aeon::session::TextDelta;

namespace {

uint32_t checks = 0;
uint32_t failures = 0;

void expect(bool ok, const std::string& label) {
    ++checks;
    std::printf("  %-66s %s\n", label.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

class FakeEngine : public ConversationEngine {
public:
    using Handler = std::function<ConversationResult(
        const ConversationRequest&, const DeltaSink&, const std::function<bool()>&)>;

    ConversationInfo info() const override {
        ConversationInfo i;
        i.model_id = "fake-model";
        i.context_capacity = 512;
        i.resident_tokens = 0;
        i.invariants_ok = true;
        return i;
    }

    ConversationResult run(const ConversationRequest& request, const DeltaSink& sink,
                           const std::function<bool()>& cancelled) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_request_ = request;
        }
        if (handler) return handler(request, sink, cancelled);
        ConversationResult r;
        r.stop = aeon::text::StopReason::Eos;
        return r;
    }

    void reset() noexcept override { ++reset_count_; }

    ConversationRequest last_request() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_request_;
    }
    int reset_count() const { return reset_count_.load(); }
    bool saw_cancel() const { return saw_cancel_.load(); }
    void note_cancel() { saw_cancel_.store(true); }

    Handler handler;
    std::atomic<int> reset_count_{0};
    std::atomic<bool> saw_cancel_{false};

private:
    mutable std::mutex mutex_;
    ConversationRequest last_request_;
};

// --- SSE helpers -------------------------------------------------------------

std::string sse_content(const std::string& raw) {
    std::string out;
    size_t position = 0;
    while ((position = raw.find("data: ", position)) != std::string::npos) {
        const size_t end = raw.find('\n', position);
        const std::string line = raw.substr(position + 6, end - position - 6);
        position = end == std::string::npos ? raw.size() : end;
        if (line == "[DONE]") continue;
        try {
            const JsonValue value = JsonValue::parse(line);
            const JsonValue* choices = value.find("choices");
            if (choices == nullptr || !choices->is_array() || choices->as_array().empty()) continue;
            const JsonValue* delta = choices->as_array()[0].find("delta");
            if (delta == nullptr || !delta->is_object()) continue;
            if (const JsonValue* content = delta->find("content");
                content != nullptr && content->is_string()) {
                out += content->as_string();
            }
        } catch (const std::exception&) {
        }
    }
    return out;
}

bool sse_has_done(const std::string& raw) { return raw.find("data: [DONE]") != std::string::npos; }

std::string stream_request(const std::string& text) {
    return "{\"messages\":[{\"role\":\"user\",\"content\":\"" + text +
           "\"}],\"stream\":true}";
}

struct Fixture {
    std::shared_ptr<FakeEngine> engine = std::make_shared<FakeEngine>();
    std::unique_ptr<ConversationService> service;
    std::unique_ptr<HttpServer> http;
    std::thread worker;
    bool worker_started{false};

    explicit Fixture(size_t max_queue = 16) {
        service = std::make_unique<ConversationService>(max_queue);
        HttpOptions options;
        options.host = "127.0.0.1";
        options.port = 0;
        options.max_queue = max_queue;
        http = std::make_unique<HttpServer>(*service, options);
        http->start();
    }

    void start_worker() {
        worker = std::thread([this] { service->run_worker(*engine); });
        worker_started = true;
        for (int i = 0; i < 2000 && service->state() != ConversationService::State::Ready; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int port() const { return http->port(); }

    ~Fixture() {
        http->stop();
        if (worker_started) {
            service->shutdown();
            if (worker.joinable()) worker.join();
        }
    }
};

httplib::Result post_json(httplib::Client& client, const std::string& path,
                          const std::string& body) {
    return client.Post(path.c_str(), body, "application/json");
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  http layer\n");
    std::printf("================================================================================\n");

    // --- R6: /health is 503 before the worker is ready, then 200 ---------------
    {
        Fixture fixture;
        httplib::Client client("127.0.0.1", fixture.port());
        const auto health_before = client.Get("/health");
        expect(health_before && health_before->status == 503,
               "R6: /health is 503 before the worker is ready");
        fixture.start_worker();
        const auto health_after = client.Get("/health");
        expect(health_after && health_after->status == 200, "R6: /health is 200 once ready");
        const auto models = client.Get("/v1/models");
        expect(models && models->status == 200 &&
                   models->body.find("fake-model") != std::string::npos,
               "R6: /v1/models lists the model");
    }

    // --- R1/R8: non-stream equals the deltas; request matches the body ---------
    {
        Fixture fixture;
        fixture.engine->handler = [](const ConversationRequest&, const DeltaSink& sink,
                                     const std::function<bool()>&) {
            sink(TextDelta{Channel::Reasoning, "think "});
            sink(TextDelta{Channel::Content, "hello "});
            sink(TextDelta{Channel::Content, "world"});
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            r.completion_tokens = 3;
            r.prompt_tokens = 1;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());

        const std::string body =
            R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"z"}}]})";
        const auto response = post_json(client, "/v1/chat/completions", body);
        expect(response && response->status == 200, "R1: non-stream returns 200");
        const JsonValue value = JsonValue::parse(response->body);
        const JsonValue& message = value.at("choices").as_array()[0].at("message");
        expect(message.at("content").as_string() == "hello world",
               "R1: content equals the concatenated content deltas");
        expect(message.at("reasoning_content").as_string() == "think ",
               "R1: reasoning_content carries the reasoning deltas");
        const ConversationRequest received = fixture.engine->last_request();
        expect(received.tools_json == R"([{"type":"function","function":{"name":"z"}}])",
               "R8: the raw tools bytes reach the engine unchanged");
    }

    // --- R2: streaming is not buffered (the latch case) ------------------------
    {
        Fixture fixture;
        std::mutex gate_mutex;
        std::condition_variable gate_cv;
        bool client_saw_first = false;

        fixture.engine->handler = [&](const ConversationRequest&, const DeltaSink& sink,
                                      const std::function<bool()>&) {
            sink(TextDelta{Channel::Content, "one"});
            // Block until the client has actually received the first chunk. A
            // buffering server never flushes it, so this would time out.
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return client_saw_first; });
            sink(TextDelta{Channel::Content, "two"});
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            r.completion_tokens = 2;
            return r;
        };
        fixture.start_worker();

        httplib::Client client("127.0.0.1", fixture.port());
        std::string raw;
        auto receiver = [&](const char* data, size_t length) {
            raw.append(data, length);
            if (!client_saw_first && raw.find("\"content\":\"one\"") != std::string::npos) {
                std::lock_guard<std::mutex> lock(gate_mutex);
                client_saw_first = true;
                gate_cv.notify_all();
            }
            return true;
        };
        const auto response = client.Post("/v1/chat/completions", httplib::Headers{},
                                          stream_request("hi"), "application/json", receiver);
        expect(response && response->status == 200, "R2: stream returns 200");
        expect(sse_content(raw) == "onetwo", "R2: streamed deltas concatenate to the full text");
        expect(sse_has_done(raw), "R2: the stream ends with [DONE]");
    }

    // --- R3: a second request queues, does not interleave ----------------------
    {
        Fixture fixture;
        std::mutex gate_mutex;
        std::condition_variable gate_cv;
        std::atomic<int> concurrency{0};
        std::atomic<int> max_concurrency{0};
        bool release = false;

        fixture.engine->handler = [&](const ConversationRequest&, const DeltaSink& sink,
                                      const std::function<bool()>&) {
            const int now = concurrency.fetch_add(1) + 1;
            int observed = max_concurrency.load();
            while (observed < now && !max_concurrency.compare_exchange_weak(observed, now)) {}
            sink(TextDelta{Channel::Content, "x"});
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return release; });
            }
            concurrency.fetch_sub(1);
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());
        httplib::Client client_b("127.0.0.1", fixture.port());

        std::thread first([&] {
            (void)client.Post("/v1/chat/completions", httplib::Headers{}, stream_request("a"),
                              "application/json", [](const char*, size_t) { return true; });
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string second_raw;
        std::mutex raw_mutex;
        std::condition_variable raw_cv;
        std::thread second([&] {
            (void)client_b.Post("/v1/chat/completions", httplib::Headers{}, stream_request("b"),
                                "application/json", [&](const char* data, size_t length) {
                                    {
                                        std::lock_guard<std::mutex> lock(raw_mutex);
                                        second_raw.append(data, length);
                                    }
                                    raw_cv.notify_all();
                                    return true;
                                });
        });
        {
            std::unique_lock<std::mutex> lock(raw_mutex);
            raw_cv.wait_for(lock, std::chrono::seconds(5), [&] {
                return second_raw.find(": queued position=1") != std::string::npos;
            });
        }
        std::string second_snapshot;
        {
            std::lock_guard<std::mutex> lock(raw_mutex);
            second_snapshot = second_raw;
        }
        // The queued request has seen only keep-alive comments, no data.
        expect(second_snapshot.find(": queued position=1") != std::string::npos,
               "R3: a queued stream receives `: queued position=1`");
        expect(sse_content(second_snapshot).empty(), "R3: a queued stream has no text yet");

        {
            std::lock_guard<std::mutex> lock(gate_mutex);
            release = true;
        }
        gate_cv.notify_all();
        first.join();
        second.join();
        expect(max_concurrency.load() == 1, "R3: the engine never ran two jobs at once");
    }

    // --- R4: a client that leaves is cancelled --------------------------------
    {
        Fixture fixture;
        fixture.engine->handler = [&](const ConversationRequest&, const DeltaSink& sink,
                                      const std::function<bool()>& cancelled) {
            sink(TextDelta{Channel::Content, "one"});
            for (int i = 0; i < 300 && !cancelled(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                sink(TextDelta{Channel::Content, "x"});
            }
            if (cancelled()) fixture.engine->note_cancel();
            ConversationResult r;
            r.stop = cancelled() ? aeon::text::StopReason::Cancelled : aeon::text::StopReason::Eos;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());
        bool saw_first = false;
        const auto response = client.Post(
            "/v1/chat/completions", httplib::Headers{}, stream_request("a"), "application/json",
            [&](const char* data, size_t length) {
                (void)data;
                (void)length;
                if (!saw_first) {
                    // Abort the download once the first chunk has arrived: the
                    // client closes the socket.
                    saw_first = true;
                    return false;
                }
                return true;
            });
        (void)response;
        // Give the server a moment to notice the disconnect and cancel.
        for (int i = 0; i < 400 && !fixture.engine->saw_cancel(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        expect(fixture.engine->saw_cancel(), "R4: closing the client sets the cancel flag");

        // The next request is served normally.
        httplib::Client client2("127.0.0.1", fixture.port());
        fixture.engine->handler = [](const ConversationRequest&, const DeltaSink&,
                                     const std::function<bool()>&) {
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            return r;
        };
        const auto next = post_json(client2, "/v1/chat/completions",
                                    R"({"messages":[{"role":"user","content":"hi"}]})");
        expect(next && next->status == 200, "R4: the next request is served after a cancel");
    }

    // --- R5: defined errors ----------------------------------------------------
    {
        Fixture fixture;
        fixture.engine->handler = [](const ConversationRequest& request, const DeltaSink&,
                                     const std::function<bool()>&) {
            const std::string content =
                request.messages.empty() ? "" : request.messages.front().content;
            if (content == "overflow") {
                throw ConversationError(ErrorKind::ContextOverflow, "prompt too long");
            }
            if (content == "boom") {
                throw ConversationError(ErrorKind::Internal, "engine failed");
            }
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());

        auto bad = post_json(client, "/v1/chat/completions", "{not json}");
        expect(bad && bad->status == 400, "R5: malformed JSON is 400");

        auto wrong = post_json(client, "/v1/chat/completions",
                               R"({"messages":[{"role":"wizard","content":"x"}]})");
        expect(wrong && wrong->status == 400, "R5: wrong types are 400");

        auto overflow = post_json(client, "/v1/chat/completions",
                                  R"({"messages":[{"role":"user","content":"overflow"}]})");
        expect(overflow && overflow->status == 400 &&
                   overflow->body.find("context_length_exceeded") != std::string::npos,
               "R5: context overflow is 400 context_length_exceeded");

        const int resets_before_boom = fixture.engine->reset_count();
        auto boom = post_json(client, "/v1/chat/completions",
                              R"({"messages":[{"role":"user","content":"boom"}]})");
        expect(boom && boom->status == 500, "R5: an engine failure is 500");
        expect(fixture.engine->reset_count() == resets_before_boom + 1,
               "R5: an engine failure resets the session");

        auto missing = client.Get("/no/such/route");
        expect(missing && missing->status == 404 &&
                   missing->body.find("\"error\"") != std::string::npos,
               "R5: an unknown route is a JSON 404");

        auto wrong_method = client.Get("/v1/chat/completions");
        expect(wrong_method && wrong_method->status == 404 &&
                   wrong_method->body.find("\"error\"") != std::string::npos,
               "R5: a wrong method is a JSON error");

        auto again = post_json(client, "/v1/chat/completions",
                               R"({"messages":[{"role":"user","content":"ok"}]})");
        expect(again && again->status == 200, "R5: the server answers the next request");
    }

    // --- R5: queue-full is 503 + Retry-After ----------------------------------
    {
        Fixture fixture(1);  // cap 1
        std::mutex gate_mutex;
        std::condition_variable gate_cv;
        bool release = false;
        fixture.engine->handler = [&](const ConversationRequest&, const DeltaSink&,
                                      const std::function<bool()>&) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return release; });
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());

        std::thread live([&] {
            (void)post_json(client, "/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"live"}]})");
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        httplib::Client client2("127.0.0.1", fixture.port());
        // One queued job fits (cap 1); the next is refused.
        std::thread queued([&] {
            (void)post_json(client2, "/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"q"}]})");
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        httplib::Client client3("127.0.0.1", fixture.port());
        auto full = post_json(client3, "/v1/chat/completions",
                              R"({"messages":[{"role":"user","content":"x"}]})");
        expect(full && full->status == 503 && full->has_header("Retry-After"),
               "R5: a full queue is 503 with Retry-After");

        {
            std::lock_guard<std::mutex> lock(gate_mutex);
            release = true;
        }
        gate_cv.notify_all();
        live.join();
        queued.join();
    }

    // --- R6: /status answers while the fake is blocked -------------------------
    {
        Fixture fixture;
        std::mutex gate_mutex;
        std::condition_variable gate_cv;
        bool release = false;
        fixture.engine->handler = [&](const ConversationRequest&, const DeltaSink&,
                                      const std::function<bool()>&) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return release; });
            ConversationResult r;
            r.stop = aeon::text::StopReason::Eos;
            return r;
        };
        fixture.start_worker();
        httplib::Client client("127.0.0.1", fixture.port());

        std::thread blocked([&] {
            (void)post_json(client, "/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"live"}]})");
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        httplib::Client status_client("127.0.0.1", fixture.port());
        const auto status = status_client.Get("/status");
        expect(status && status->status == 200, "R6: /status answers while the engine is blocked");
        if (status) {
            const JsonValue value = JsonValue::parse(status->body);
            expect(value.at("state").as_string() == "ready", "R6: /status reports the state");
            const JsonValue* live = value.find("live");
            expect(live != nullptr && live->is_object(), "R6: /status reports the live job");
        }

        {
            std::lock_guard<std::mutex> lock(gate_mutex);
            release = true;
        }
        gate_cv.notify_all();
        blocked.join();
    }

    std::printf("================================================================================\n");
    std::printf("  %u/%u checks passed\n", checks - failures, checks);
    std::printf("================================================================================\n");
    return failures == 0 ? 0 : 1;
}
