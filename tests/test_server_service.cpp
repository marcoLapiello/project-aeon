// -----------------------------------------------------------------------------
// Gate — the conversation service.
//
// The service is the concurrency seam: HTTP threads `submit` and block on a job's
// stream, one worker runs jobs FIFO on the engine thread. This gate drives it with
// a fake engine (scripted deltas, a latch, a re-entrancy counter, a recorded reset
// count) and pins the properties the design rests on: one run at a time, FIFO, a
// second job queues rather than being refused, a full queue is a defined refusal, a
// cancelled queued job never reaches the engine, a failed job resets the engine and
// the next one still runs, `shutdown` terminates every stream, and `snapshot` never
// waits on the engine.
//
// The fake links no HIP: this test is built by `aeon_add_host_test`, so if the
// service ever reached an engine type it would not link.
// -----------------------------------------------------------------------------

#include "server/conversation_service.hpp"

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

using aeon::server::ConversationService;
using aeon::server::JobEvent;
using aeon::server::JobStream;
using aeon::server::QueueFull;
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
    std::printf("  %-64s %s\n", label.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

class FakeEngine : public ConversationEngine {
public:
    ConversationInfo info() const override {
        ConversationInfo i;
        i.model_id = "fake";
        i.context_capacity = 128;
        i.resident_tokens = 0;
        i.invariants_ok = true;
        return i;
    }

    ConversationResult run(const ConversationRequest& request, const DeltaSink& sink,
                           const std::function<bool()>& cancelled) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++concurrency_;
            if (concurrency_ > max_concurrency.load()) max_concurrency.store(concurrency_);
            ran_.push_back(request.messages.empty() ? std::string("<empty>")
                                                     : request.messages.front().content);
        }
        for (const TextDelta& delta : deltas) sink(delta);

        bool was_cancelled = false;
        if (block_) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!released_) {
                if (cancelled && cancelled()) {
                    was_cancelled = true;
                    saw_cancel_.store(true);
                    break;
                }
                condvar_.wait_for(lock, std::chrono::milliseconds(5));
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --concurrency_;
        }

        if (was_cancelled) {
            ConversationResult r;
            r.stop = aeon::text::StopReason::Cancelled;
            return r;
        }
        if (throw_) throw ConversationError(ErrorKind::Internal, "injected failure");

        ConversationResult r;
        r.stop = aeon::text::StopReason::Eos;
        r.completion_tokens = static_cast<uint32_t>(deltas.size());
        return r;
    }

    void reset() noexcept override { ++reset_count_; }

    // --- test controls ------------------------------------------------------
    void block(bool value) {
        std::lock_guard<std::mutex> lock(mutex_);
        block_ = value;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condvar_.notify_all();
    }
    void set_deltas(std::vector<TextDelta> value) { deltas = std::move(value); }
    void set_throw(bool value) { throw_ = value; }

    std::atomic<int> max_concurrency{0};
    std::vector<std::string> ran() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ran_;
    }
    int reset_count() const { return reset_count_.load(); }
    bool saw_cancel() const { return saw_cancel_.load(); }

private:
    std::mutex mutex_;
    std::condition_variable condvar_;
    int concurrency_{0};
    bool block_{false};
    bool released_{false};
    bool throw_{false};
    std::vector<std::string> ran_;
    std::vector<TextDelta> deltas;
    std::atomic<int> reset_count_{0};
    std::atomic<bool> saw_cancel_{false};
};

ConversationRequest request_with(const std::string& marker) {
    ConversationRequest request;
    aeon::session::Message message;
    message.role = aeon::session::Role::User;
    message.content = marker;
    request.messages.push_back(message);
    return request;
}

// Drains a stream until it closes, returning the concatenated content and whether
// the terminal event was Finished.
struct Drained {
    std::string content;
    bool finished{false};
    bool failed{false};
    aeon::text::StopReason stop{aeon::text::StopReason::Error};
};

Drained drain(const std::shared_ptr<JobStream>& stream) {
    Drained out;
    JobEvent event;
    while (stream->pop(event, std::chrono::milliseconds(2000))) {
        if (event.kind == JobEvent::Kind::Delta) {
            if (event.delta.channel == Channel::Content) out.content += event.delta.text;
        } else if (event.kind == JobEvent::Kind::Finished) {
            out.finished = true;
            out.stop = event.result.stop;
            break;
        } else {
            out.failed = true;
            out.stop = aeon::text::StopReason::Error;
            break;
        }
    }
    return out;
}

void wait_ready(const ConversationService& service) {
    for (int i = 0; i < 2000 && service.state() != ConversationService::State::Ready; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Waits until the worker has picked `job` up as the live job.
void wait_live(const ConversationService& service, const std::shared_ptr<aeon::server::Job>& job) {
    for (int i = 0; i < 2000 && service.position(job) != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void wait_joined(std::thread& thread) {
    if (thread.joinable()) thread.join();
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  conversation service\n");
    std::printf("================================================================================\n");

    // --- one run at a time, FIFO, a second job queues -------------------------
    {
        FakeEngine engine;
        engine.block(true);
        engine.set_deltas({{Channel::Content, "hi"}});
        ConversationService service(16);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto first = service.submit(request_with("first"));
        wait_live(service, first);
        auto second = service.submit(request_with("second"));

        // The first is live (0); the second is the 1st waiting.
        expect(service.position(first) == 0, "first job is live (position 0)");
        expect(service.position(second) == 1, "second job is waiting (position 1)");

        // Snapshot is readable while the fake is blocked inside run.
        const auto mid = service.snapshot();
        expect(mid.state == "ready", "snapshot state is ready while blocked");
        expect(mid.live.has_value() && mid.live->id == first->id,
               "snapshot reports the live job while blocked");
        expect(mid.queue_depth == 1, "snapshot reports queue depth 1 while blocked");

        engine.release();
        const Drained first_result = drain(first->stream);
        const Drained second_result = drain(second->stream);
        expect(first_result.finished && first_result.content == "hi", "first job finished");
        expect(second_result.finished, "second job was served, not refused");

        service.shutdown();
        wait_joined(worker);
        expect(engine.max_concurrency == 1, "the engine never ran two jobs at once");
        const std::vector<std::string> order = engine.ran();
        expect(order.size() == 2 && order[0] == "first" && order[1] == "second",
               "jobs ran in FIFO order");
    }

    // --- a full queue is a defined refusal ------------------------------------
    {
        FakeEngine engine;
        engine.block(true);
        ConversationService service(2);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto live = service.submit(request_with("live"));
        wait_live(service, live);
        (void)service.submit(request_with("q1"));
        (void)service.submit(request_with("q2"));

        bool queue_full = false;
        try {
            (void)service.submit(request_with("overflow"));
        } catch (const QueueFull&) {
            queue_full = true;
        }
        expect(queue_full, "submit past the cap throws QueueFull");

        engine.release();
        drain(live->stream);
        service.shutdown();
        wait_joined(worker);
    }

    // --- a cancelled queued job never reaches the engine ----------------------
    {
        FakeEngine engine;
        engine.block(true);
        ConversationService service(16);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto live = service.submit(request_with("live"));
        wait_live(service, live);
        auto queued = service.submit(request_with("queued"));
        service.cancel(queued);
        engine.release();

        drain(live->stream);
        const Drained queued_result = drain(queued->stream);
        expect(queued_result.finished &&
                   queued_result.stop == aeon::text::StopReason::Cancelled,
               "cancelled queued job finishes Cancelled");
        const std::vector<std::string> order = engine.ran();
        expect(order.size() == 1 && order[0] == "live",
               "cancelled queued job never reached the engine");

        service.shutdown();
        wait_joined(worker);
    }

    // --- cancelling the live job is seen by the engine ------------------------
    {
        FakeEngine engine;
        engine.block(true);
        ConversationService service(16);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto live = service.submit(request_with("live"));
        wait_live(service, live);
        service.cancel(live);
        const Drained result = drain(live->stream);
        expect(result.stop == aeon::text::StopReason::Cancelled,
               "cancelling the live job stops it with Cancelled");
        expect(engine.saw_cancel(), "the engine's cancel flag was observed");

        // The next job runs clean on the same service.
        engine.block(false);
        auto next = service.submit(request_with("next"));
        const Drained next_result = drain(next->stream);
        expect(next_result.finished, "the next job runs clean after a cancel");

        service.shutdown();
        wait_joined(worker);
    }

    // --- a throwing engine fails the job, resets, and the next runs -----------
    {
        FakeEngine engine;
        engine.set_throw(true);
        ConversationService service(16);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto bad = service.submit(request_with("bad"));
        const Drained bad_result = drain(bad->stream);
        expect(bad_result.failed, "a throwing engine yields a Failed event");
        expect(engine.reset_count() == 1, "a failed job resets the engine");

        engine.set_throw(false);
        auto good = service.submit(request_with("good"));
        const Drained good_result = drain(good->stream);
        expect(good_result.finished, "the next job runs after a failure");

        service.shutdown();
        wait_joined(worker);
    }

    // --- shutdown terminates the live and queued streams ----------------------
    {
        FakeEngine engine;
        engine.block(true);
        ConversationService service(16);
        std::thread worker([&] { service.run_worker(engine); });
        wait_ready(service);

        auto live = service.submit(request_with("live"));
        wait_live(service, live);
        auto queued = service.submit(request_with("queued"));
        service.shutdown();

        const Drained live_result = drain(live->stream);
        const Drained queued_result = drain(queued->stream);
        expect(live_result.finished, "shutdown terminates the live stream");
        expect(queued_result.finished, "shutdown terminates the queued stream");
        wait_joined(worker);
        expect(service.state() == ConversationService::State::Stopping,
               "service state is stopping after shutdown");
    }

    std::printf("================================================================================\n");
    std::printf("  %u/%u checks passed\n", checks - failures, checks);
    std::printf("================================================================================\n");
    return failures == 0 ? 0 : 1;
}
