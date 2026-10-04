#include "server/conversation_service.hpp"

#include <functional>
#include <utility>

namespace aeon::server {

namespace {
using Clock = std::chrono::steady_clock;

session::ConversationResult cancelled_result() {
    session::ConversationResult result;
    result.stop = text::StopReason::Cancelled;
    return result;
}
}  // namespace

std::string ConversationService::state_name() const {
    switch (state_.load()) {
        case State::Loading: return "loading";
        case State::Ready: return "ready";
        case State::Stopping: return "stopping";
    }
    return "loading";
}

std::shared_ptr<Job> ConversationService::submit(session::ConversationRequest request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.load() != State::Ready) throw NotReady();
    if (queue_.size() >= max_queue_) throw QueueFull();

    auto job = std::make_shared<Job>();
    job->id = next_id_++;
    job->request = std::move(request);
    job->stream = std::make_shared<JobStream>();
    job->enqueued = Clock::now();
    queue_.push_back(job);
    condvar_.notify_all();
    return job;
}

void ConversationService::cancel(const std::shared_ptr<Job>& job) noexcept {
    if (job) job->cancel.store(true);
}

size_t ConversationService::position(const std::shared_ptr<Job>& job) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (live_job_ && live_job_ == job) return 0;
    size_t index = 1;
    for (const auto& queued : queue_) {
        if (queued == job) return index;
        ++index;
    }
    return 0;
}

void ConversationService::run_worker(session::ConversationEngine& engine) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        info_ = engine.info();
        state_.store(State::Ready);
    }
    condvar_.notify_all();

    for (;;) {
        std::shared_ptr<Job> job;
        double queue_wait_ms = 0.0;
        Clock::time_point started{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condvar_.wait(lock, [this] { return !running_.load() || !queue_.empty(); });
            if (!running_.load()) break;

            job = queue_.front();
            queue_.pop_front();
            started = Clock::now();
            queue_wait_ms =
                std::chrono::duration<double, std::milli>(started - job->enqueued).count();

            live_ = LiveJob{job->id, "prefilling", 0, 0.0};
            live_job_ = job;
            live_started_ = started;
            live_tokens_.store(0);
            live_seen_delta_.store(false);
        }

        session::ConversationResult result = cancelled_result();

        if (job->cancel.load()) {
            // Cancelled while queued: never reaches the engine, so the live session
            // is untouched.
            job->stream->push(JobEvent::make_finished(result));
        } else {
            const session::DeltaSink sink = [this, job](const session::TextDelta& delta) {
                // The first delta marks the switch from prefill to decode in the
                // snapshot's phase.
                live_seen_delta_.store(true);
                live_tokens_.fetch_add(1);
                job->stream->push(JobEvent::make_delta(delta));
            };
            const std::function<bool()> cancelled = [job] { return job->cancel.load(); };
            try {
                result = engine.run(job->request, sink, cancelled);
                job->stream->push(JobEvent::make_finished(result));
            } catch (const session::ConversationError& error) {
                engine.reset();
                result.stop = text::StopReason::Error;
                job->stream->push(JobEvent::make_failed(error.kind, error.what()));
            } catch (const std::exception& error) {
                engine.reset();
                result.stop = text::StopReason::Error;
                job->stream->push(
                    JobEvent::make_failed(session::ErrorKind::Internal, error.what()));
            }
        }
        // Exactly one terminal event was pushed above; closing lets a drained
        // consumer stop waiting.
        job->stream->close();

        const session::ConversationInfo info = engine.info();
        const double total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            info_ = info;
            last_ = LastRun{result, queue_wait_ms, total_ms};
            live_.reset();
            live_job_.reset();
        }
    }

    // Any job still queued when the worker stops is failed with Cancelled. (A
    // well-behaved `shutdown` already drained the queue; this is the other order.)
    std::deque<std::shared_ptr<Job>> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.swap(queue_);
    }
    for (const auto& queued : pending) {
        queued->stream->push(JobEvent::make_finished(cancelled_result()));
        queued->stream->close();
    }
    state_.store(State::Stopping);
}

void ConversationService::shutdown() noexcept {
    std::deque<std::shared_ptr<Job>> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(false);
        state_.store(State::Stopping);
        if (live_job_) live_job_->cancel.store(true);
        pending.swap(queue_);
    }
    condvar_.notify_all();
    for (const auto& job : pending) {
        job->stream->push(JobEvent::make_finished(cancelled_result()));
        job->stream->close();
    }
}

ServerSnapshot ConversationService::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ServerSnapshot snap;
    snap.state = state_name();
    snap.queue_depth = queue_.size();
    snap.info = info_;
    if (last_) snap.last = last_;
    if (live_) {
        LiveJob live = *live_;
        live.tokens_so_far = live_tokens_.load();
        live.phase = live_seen_delta_.load() ? "decoding" : "prefilling";
        live.elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - live_started_).count();
        snap.live = live;
    }
    return snap;
}

}  // namespace aeon::server
