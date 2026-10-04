#pragma once

// -----------------------------------------------------------------------------
// ConversationService — the one place the engine thread and the HTTP threads meet.
//
// The engine is driven on exactly one thread (HIP's device context is per-thread),
// so requests cannot run on an HTTP handler. Instead: HTTP handlers `submit` a job
// and block on its `JobStream`; a single worker thread runs one job at a time,
// FIFO, and pushes events. Nothing in this file names a HIP, model, weight-format
// or kernel type — it drives `session::ConversationEngine`, the neutral seam.
//
// The queue is a fairness policy: a second request waits, it is not refused. Only
// the service state (loading / ready / stopping) and a full queue will refuse, and
// both are defined errors the HTTP layer turns into a 503.
// -----------------------------------------------------------------------------

#include "infrastructure/session/conversation.hpp"
#include "server/job_stream.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace aeon::server {

// Thrown by `submit`. The HTTP layer maps both to 503 + Retry-After.
struct QueueFull : std::runtime_error {
    QueueFull() : std::runtime_error("the request queue is full") {}
};

struct NotReady : std::runtime_error {
    NotReady() : std::runtime_error("the engine is not ready") {}
};

struct Job {
    uint64_t id{0};
    session::ConversationRequest request;
    std::shared_ptr<JobStream> stream;
    std::atomic<bool> cancel{false};
    std::chrono::steady_clock::time_point enqueued{};
};

// What `/status` reports about the run currently on the engine. `phase` is
// "prefilling" until the first delta arrives and "decoding" after — the closest
// the neutral seam can get to the engine's own two strategies without naming them.
struct LiveJob {
    uint64_t id{0};
    std::string phase{"loading"};
    uint64_t tokens_so_far{0};
    double elapsed_ms{0.0};
};

struct LastRun {
    session::ConversationResult result;
    double queue_wait_ms{0.0};
    double total_ms{0.0};
};

struct ServerSnapshot {
    std::string state{"loading"};
    size_t queue_depth{0};
    std::optional<LiveJob> live;
    session::ConversationInfo info;
    std::optional<LastRun> last;
};

class ConversationService {
public:
    enum class State { Loading, Ready, Stopping };

    explicit ConversationService(size_t max_queue) : max_queue_(max_queue) {}

    // Enqueue a job. Throws `NotReady` unless the worker has published `Ready`,
    // or `QueueFull` when the queue is at capacity.
    std::shared_ptr<Job> submit(session::ConversationRequest request);

    // Ask the job to stop. For a live job this is the flag the engine's `cancelled`
    // hook polls; for a queued job the worker skips it before it reaches the engine.
    void cancel(const std::shared_ptr<Job>& job) noexcept;

    // 0 when the job is live; n (1-based) when it is the n-th waiting job; 0 when it
    // is neither.
    size_t position(const std::shared_ptr<Job>& job) const;

    // Drive the engine until `shutdown`. Runs one job at a time and blocks; the
    // caller runs it on the engine thread.
    void run_worker(session::ConversationEngine& engine);

    // Refuse new work, cancel the live job, fail the queued ones, and unblock
    // `run_worker`.
    void shutdown() noexcept;

    // A copy under the lock, readable from any thread without waiting on the
    // engine.
    ServerSnapshot snapshot() const;

    State state() const noexcept { return state_.load(); }

private:
    std::string state_name() const;

    size_t max_queue_;
    mutable std::mutex mutex_;
    std::condition_variable_any condvar_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::atomic<State> state_{State::Loading};
    std::atomic<bool> running_{true};
    uint64_t next_id_{1};

    // Live-run bookkeeping, written by the worker and read by `snapshot`. `live_`
    // and `last_` are guarded by `mutex_`; the two counters are atomic because the
    // delta sink bumps them without taking the lock.
    std::optional<LiveJob> live_;
    std::optional<LastRun> last_;
    std::chrono::steady_clock::time_point live_started_{};
    std::atomic<uint64_t> live_tokens_{0};
    std::atomic<bool> live_seen_delta_{false};
    std::shared_ptr<Job> live_job_;
    session::ConversationInfo info_{};
};

}  // namespace aeon::server
