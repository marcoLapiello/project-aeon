#pragma once

// -----------------------------------------------------------------------------
// JobStream — one job's event queue, shared between the worker thread and the
// HTTP thread that is draining it.
//
// Exactly one terminal event (Finished or Failed) is ever pushed, and `close` is
// what lets a consumer waiting on `pop` return instead of blocking forever. The
// stream is created by `submit` and shared by `shared_ptr`; the worker owns the
// push side, one HTTP handler owns the pop side.
//
// It is intentionally tiny and knows nothing of sessions or HTTP: it carries the
// neutral seam's own types (`TextDelta`, `ConversationResult`, `ErrorKind`).
// -----------------------------------------------------------------------------

#include "infrastructure/session/conversation.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace aeon::server {

struct JobEvent {
    enum class Kind { Delta, Finished, Failed };

    Kind kind{Kind::Delta};
    session::TextDelta delta;                 // Kind::Delta
    session::ConversationResult result;       // Kind::Finished
    session::ErrorKind error_kind{session::ErrorKind::Internal};  // Kind::Failed
    std::string error_message;                // Kind::Failed

    static JobEvent make_delta(session::TextDelta value) {
        JobEvent event;
        event.kind = Kind::Delta;
        event.delta = std::move(value);
        return event;
    }
    static JobEvent make_finished(session::ConversationResult value) {
        JobEvent event;
        event.kind = Kind::Finished;
        event.result = std::move(value);
        return event;
    }
    static JobEvent make_failed(session::ErrorKind kind, std::string message) {
        JobEvent event;
        event.kind = Kind::Failed;
        event.error_kind = kind;
        event.error_message = std::move(message);
        return event;
    }
};

class JobStream {
public:
    void push(JobEvent event) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            events_.push_back(std::move(event));
        }
        condvar_.notify_all();
    }

    // Marks the stream complete: once closed and drained, `pop` returns false.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        condvar_.notify_all();
    }

    // Waits up to `timeout` for an event. Returns true when one was retrieved;
    // false on timeout, or when the stream is closed and drained.
    bool pop(JobEvent& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condvar_.wait_for(lock, timeout, [this] { return !events_.empty() || closed_; })) {
            return false;  // timeout
        }
        if (events_.empty()) return false;  // closed and drained
        out = std::move(events_.front());
        events_.pop_front();
        return true;
    }

    // Whether the producer has finished with the stream. Combined with a failed
    // `pop`, this tells a timeout from a drained stream.
    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condvar_;
    std::deque<JobEvent> events_;
    bool closed_{false};
};

}  // namespace aeon::server
