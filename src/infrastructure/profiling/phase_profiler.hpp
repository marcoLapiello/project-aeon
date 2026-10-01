#pragma once

// -----------------------------------------------------------------------------
// A phase timer for the chunked layer body, so "where does prefill's time go" is
// answered by a measurement rather than by counting launches from the source.
//
// Two numbers per phase, because they answer different questions and only their
// difference is diagnostic:
//
//   * **host** — wall time spent *issuing* the phase, measured between the two
//     `region` calls. This is the cost the CPU pays to submit the work, and it is
//     what an issuance-bound path is actually spending its time on.
//   * **gpu** — `hipEvent` elapsed time between the same two points on the phase's
//     stream. This is the work the device did inside the phase.
//
// If the host column dominates, the path is issuance-bound and the lever is fewer,
// larger launches (batching). If the GPU column dominates, the lever is the kernel.
// A single elapsed number cannot tell the two apart, which is why the pair exists.
//
// Events are recorded asynchronously and read once in `resolve()` after the caller
// has synchronized, so a phase boundary costs two event records, not a stall — the
// instrumentation must not create the synchronisation it is trying to find.
//
// A disabled profiler is a branch; `Region` is a no-op when the profiler is off.
// -----------------------------------------------------------------------------

#include "infrastructure/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace aeon::core {

class PhaseProfiler {
public:
    using Clock = std::chrono::steady_clock;

    static PhaseProfiler& instance() {
        static PhaseProfiler profiler;
        return profiler;
    }

    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
    bool enabled() const noexcept { return enabled_; }

    // A per-name sample stride. At per-row granularity the two event records per
    // region cost more than the region measures, so a region finer than the phase
    // can only be timed by sampling it. A stride of `n` records one occurrence in
    // every `n` and reports the mean scaled back to the full count, which keeps the
    // total comparable to an unsampled phase. `1` (the default) records everything.
    void set_sample_stride(size_t stride) noexcept {
        sample_stride_ = stride == 0 ? 1 : stride;
    }

    // Drops every sample and destroys its events, so a previous window cannot leak
    // into the next report.
    void reset() {
        for (Sample& sample : samples_) {
            for (auto& span : sample.spans) {
                destroy(span.first);
                destroy(span.second);
            }
            sample.spans.clear();
        }
        samples_.clear();
        resolved_ = false;
    }

    // A named region. Move-only; the destructor closes it.
    //
    // The host timing measures the *issuing* of the phase, which is the number that
    // separates an issuance-bound path from a compute-bound one. It is the span
    // between the two event records; the GPU span is what the events themselves
    // read, so the two are directly comparable.
    class Region {
    public:
        Region() = default;
        Region(PhaseProfiler* profiler, const char* name, hipStream_t stream)
            : profiler_(profiler) {
            if (profiler_ == nullptr || !profiler_->enabled_ || stream == nullptr) {
                profiler_ = nullptr;
                return;
            }
            index_ = profiler_->start(name, stream);
            if (index_ == kSkipped) {
                profiler_ = nullptr;
                return;
            }
            host_start_ = Clock::now();
        }
        Region(const Region&) = delete;
        Region& operator=(const Region&) = delete;
        Region(Region&& other) noexcept { move_from(other); }
        Region& operator=(Region&& other) noexcept {
            close();
            move_from(other);
            return *this;
        }
        ~Region() { close(); }

    private:
        void move_from(Region& other) noexcept {
            profiler_ = other.profiler_;
            index_ = other.index_;
            host_start_ = other.host_start_;
            other.profiler_ = nullptr;
        }
        void close() noexcept {
            if (profiler_ != nullptr) {
                profiler_->stop(index_,
                                std::chrono::duration<double, std::milli>(
                                    Clock::now() - host_start_)
                                    .count());
                profiler_ = nullptr;
            }
        }
        PhaseProfiler* profiler_{nullptr};
        size_t index_{0};
        Clock::time_point host_start_{};
    };

    // Returned by `start` when the sample stride skips this occurrence; the `Region`
    // becomes a no-op so the caller pays neither an event nor a stop call.
    static constexpr size_t kSkipped = static_cast<size_t>(-1);

    [[nodiscard]] Region region(const char* name, hipStream_t stream) {
        return Region(this, name, stream);
    }

    // Reads every pending event. Call once the stream is idle.
    void resolve() {
        if (!enabled_) return;
        for (Sample& sample : samples_) {
            for (auto& span : sample.spans) {
                if (span.first == nullptr || span.second == nullptr) continue;
                float ms = 0.0f;
                if (hipEventSynchronize(span.second) == hipSuccess &&
                    hipEventElapsedTime(&ms, span.first, span.second) == hipSuccess) {
                    sample.gpu_ms += static_cast<double>(ms);
                }
                destroy(span.first);
                destroy(span.second);
            }
            sample.spans.clear();
        }
        resolved_ = true;
    }

    void report(std::FILE* out) const {
        if (samples_.empty()) return;
        // A region nested inside another (its name is indented) is a breakdown of a
        // parent, so summing it into the total would count the same GPU work twice.
        // The total is taken from top-level regions only; nested ones are shown for
        // their share but never added in.
        auto top_level = [](const std::string& name) {
            return name.empty() || name[0] != ' ';
        };
        double total_gpu = 0.0;
        double total_host = 0.0;
        for (const Sample& sample : samples_) {
            if (!top_level(sample.name)) continue;
            const double scale = sample.recorded > 0
                ? static_cast<double>(sample.calls) / static_cast<double>(sample.recorded)
                : 0.0;
            total_gpu += sample.gpu_ms * scale;
            total_host += sample.host_ms * scale;
        }
        std::fprintf(out, "\n  [Phase] %-30s %11s %11s %7s\n", "phase", "host ms",
                     "gpu ms", "gpu%");
        for (const Sample& sample : samples_) {
            const double scale = sample.recorded > 0
                ? static_cast<double>(sample.calls) / static_cast<double>(sample.recorded)
                : 0.0;
            std::fprintf(out, "  [Phase] %-30s %11.1f %11.1f %6.1f%%\n",
                         sample.name.c_str(), sample.host_ms * scale,
                         sample.gpu_ms * scale,
                         total_gpu > 0.0 ? 100.0 * sample.gpu_ms * scale / total_gpu : 0.0);
        }
        std::fprintf(out, "  [Phase] %-30s %11.1f %11.1f\n", "total", total_host,
                     total_gpu);
    }

private:
    struct Sample {
        std::string name;
        double host_ms{0.0};
        double gpu_ms{0.0};
        // Occurrences and how many were actually recorded. With a sample stride the
        // two differ, and `report` scales the recorded sum back by their ratio.
        uint64_t calls{0};
        uint64_t recorded{0};
        // One (start, stop) pair per closure. Kept as a list rather than overwritten
        // so a phase visited once per chunk accumulates instead of measuring only the
        // last chunk — the difference between a per-window total and one chunk.
        std::vector<std::pair<hipEvent_t, hipEvent_t>> spans;
        hipStream_t stream{nullptr};
    };

    size_t start(const char* name, hipStream_t stream) {
        size_t index = 0;
        for (; index < samples_.size(); ++index) {
            if (samples_[index].name == name) break;
        }
        if (index == samples_.size()) {
            Sample sample;
            sample.name = name;
            samples_.push_back(std::move(sample));
        }
        Sample& sample = samples_[index];
        sample.calls += 1;
        if (sample_stride_ > 1 && (sample.calls % sample_stride_) != 0) {
            return kSkipped;
        }
        sample.recorded += 1;
        sample.stream = stream;
        const hipEvent_t start_event = make_event();
        const hipEvent_t stop_event = make_event();
        CHECK_HIP(hipEventRecord(start_event, stream));
        sample.spans.emplace_back(start_event, stop_event);
        return index;
    }

    void stop(size_t index, double host_ms) {
        Sample& sample = samples_[index];
        CHECK_HIP(hipEventRecord(sample.spans.back().second, sample.stream));
        sample.host_ms += host_ms;
    }

    static hipEvent_t make_event() {
        hipEvent_t event = nullptr;
        CHECK_HIP(hipEventCreate(&event));
        return event;
    }

    size_t sample_stride_{1};

    static void destroy(hipEvent_t& event) noexcept {
        if (event != nullptr) {
            (void)hipEventDestroy(event);
            event = nullptr;
        }
    }

    bool enabled_{false};
    bool resolved_{false};
    std::vector<Sample> samples_;
};

} // namespace aeon::core
