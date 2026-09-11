#pragma once

#include "core/expert_registry.hpp"
#include "core/routing_counter.hpp"

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aeon::core {

enum class SupplyTelemetryPhase : uint8_t {
    Warmup = 0,
    Prefill = 1,
    Decode = 2
};

class SupplyTelemetry {
public:
    static constexpr uint32_t schema_version = 1;

    struct Summary {
        uint64_t request_count{0};
        uint64_t decode_token_count{0};
        uint64_t bytes_from_nvme{0};
        uint64_t bytes_from_host{0};
        uint64_t logical_bytes_from_warm{0};
        uint64_t logical_bytes_from_transient_staging{0};
        uint64_t h2d_bytes{0};
        uint64_t d2h_bytes{0};
        uint64_t nvme_read_service_ns{0};
        uint64_t nvme_completion_wait_ns{0};
        uint64_t h2d_enqueue_to_ready_ns{0};
        uint64_t gpu_readiness_wait_ns{0};
        uint64_t optional_demotion_wait_ns{0};
        uint64_t staging_wait_ns{0};
        uint64_t staging_reuse_wait_ns{0};
        uint64_t demotion_attempts{0};
        uint64_t demotion_completions{0};
        uint64_t demotion_drops{0};
        uint64_t hot_occupancy_min{0};
        uint64_t hot_occupancy_max{0};
        uint64_t warm_valid_slots_min{0};
        uint64_t warm_valid_slots_max{0};
        uint64_t pending_transfers_max{0};
        uint64_t demotion_queue_depth_max{0};
        uint64_t warm_pinned_bytes{0};
        uint64_t warm_unpinned_bytes{0};
        uint64_t rss_bytes_peak{0};
        int64_t vm_swap_bytes_delta{0};
        std::map<std::string, uint64_t> demotion_drop_reasons;
        bool occupancy_observed{false};
    };

    void enable_jsonl(const std::string& path, std::string run_id) {
        output_.open(path, std::ios::out | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error("SupplyTelemetry: failed to open JSONL output: " + path);
        }
        enabled_ = true;
        run_id_ = std::move(run_id);
        reset();
    }

    void disable() {
        flush();
        if (output_.is_open()) {
            output_.close();
        }
        enabled_ = false;
    }

    bool enabled() const {
        return enabled_;
    }

    void set_warmup(bool warmup) {
        current_phase_ = warmup ? SupplyTelemetryPhase::Warmup : current_phase_;
    }

    void set_phase(RoutingPhase phase) {
        current_phase_ = phase == RoutingPhase::Prefill
            ? SupplyTelemetryPhase::Prefill
            : SupplyTelemetryPhase::Decode;
    }

    SupplyTelemetryPhase current_phase() const {
        return current_phase_;
    }

    void reset() {
        for (auto& phase : summaries_) {
            for (auto& summary : phase) {
                summary = Summary{};
            }
        }
        transfer_events_.clear();
        boundary_id_++;
        baseline_vm_swap_bytes_ = read_vm_swap_bytes();
        dirty_ = false;
    }

    void record_request(
        SupplyTelemetryPhase phase,
        ExpertTier source_tier,
        uint64_t logical_bytes,
        uint64_t h2d_bytes,
        uint64_t nvme_bytes,
        uint64_t host_bytes,
        uint64_t transient_bytes
    ) {
        if (!enabled_) return;
        auto& summary = summary_for(phase, source_tier);
        ++summary.request_count;
        dirty_ = true;
        summary.h2d_bytes += h2d_bytes;
        summary.bytes_from_nvme += nvme_bytes;
        summary.bytes_from_host += host_bytes;
        summary.logical_bytes_from_warm += source_tier == ExpertTier::WARM_HOST ? logical_bytes : 0;
        summary.logical_bytes_from_transient_staging += transient_bytes;
    }

    void record_decode_token() {
        if (!enabled_) return;
        for (auto& summary : summaries_[static_cast<uint32_t>(current_phase_)]) {
            if (summary.request_count != 0) {
                ++summary.decode_token_count;
                dirty_ = true;
            }
        }
    }

    void record_demotion_attempt(ExpertTier source_tier) {
        if (!enabled_) return;
        summary_for(current_phase_, source_tier).demotion_attempts++;
        dirty_ = true;
    }

    void record_demotion_completion(ExpertTier source_tier, uint64_t bytes) {
        if (!enabled_) return;
        auto& summary = summary_for(current_phase_, source_tier);
        ++summary.demotion_completions;
        summary.d2h_bytes += bytes;
        dirty_ = true;
    }

    void record_demotion_drop(ExpertTier source_tier, const std::string& reason) {
        if (!enabled_) return;
        auto& summary = summary_for(current_phase_, source_tier);
        ++summary.demotion_drops;
        ++summary.demotion_drop_reasons[reason];
        dirty_ = true;
    }

    void record_timing(
        SupplyTelemetryPhase phase,
        ExpertTier source_tier,
        uint64_t nvme_read_service_ns,
        uint64_t nvme_completion_wait_ns,
        uint64_t h2d_enqueue_to_ready_ns,
        uint64_t gpu_readiness_wait_ns,
        uint64_t staging_wait_ns,
        uint64_t staging_reuse_wait_ns
    ) {
        if (!enabled_) return;
        auto& summary = summary_for(phase, source_tier);
        summary.nvme_read_service_ns += nvme_read_service_ns;
        summary.nvme_completion_wait_ns += nvme_completion_wait_ns;
        summary.h2d_enqueue_to_ready_ns += h2d_enqueue_to_ready_ns;
        summary.gpu_readiness_wait_ns += gpu_readiness_wait_ns;
        summary.staging_wait_ns += staging_wait_ns;
        summary.staging_reuse_wait_ns += staging_reuse_wait_ns;
        dirty_ = true;
    }

    void observe_occupancy(
        uint64_t hot_slots,
        uint64_t warm_slots,
        uint64_t pending_transfers,
        uint64_t demotion_queue_depth,
        uint64_t warm_pinned_bytes,
        uint64_t warm_unpinned_bytes,
        ExpertTier source_tier
    ) {
        if (!enabled_) return;
        auto& summary = summary_for(current_phase_, source_tier);
        if (summary.request_count != 0) {
            if (!summary.occupancy_observed) {
                summary.hot_occupancy_min = hot_slots;
                summary.warm_valid_slots_min = warm_slots;
                summary.occupancy_observed = true;
            } else {
                summary.hot_occupancy_min = std::min(summary.hot_occupancy_min, hot_slots);
                summary.warm_valid_slots_min = std::min(summary.warm_valid_slots_min, warm_slots);
            }
            summary.hot_occupancy_max = std::max(summary.hot_occupancy_max, hot_slots);
            summary.warm_valid_slots_max = std::max(summary.warm_valid_slots_max, warm_slots);
            summary.pending_transfers_max = std::max(summary.pending_transfers_max, pending_transfers);
            summary.demotion_queue_depth_max = std::max(summary.demotion_queue_depth_max, demotion_queue_depth);
            summary.warm_pinned_bytes = warm_pinned_bytes;
            summary.warm_unpinned_bytes = warm_unpinned_bytes;
            observe_process_memory(summary);
        }
        dirty_ = true;
    }

    void record_transfer_event(
        uint64_t transfer_id,
        uint32_t expert_id,
        const char* operation,
        int32_t source_slot,
        int32_t destination_slot,
        const char* status,
        const char* reason = nullptr
    ) {
        if (!enabled_) return;
        std::ostringstream record;
        record << "{\"record_type\":\"transfer_event\",\"schema_version\":"
               << schema_version << ",\"run_id\":\"" << run_id_
               << "\",\"transfer_id\":" << transfer_id
               << ",\"expert_id\":" << expert_id
               << ",\"operation\":\"" << operation
               << "\",\"source_slot\":" << source_slot
               << ",\"destination_slot\":" << destination_slot
               << ",\"status\":\"" << status << "\"";
        if (reason != nullptr) {
            record << ",\"reason\":\"" << reason << "\"";
        }
        record << "}";
        transfer_events_.push_back(record.str());
        dirty_ = true;
    }

    void flush() {
        if (!enabled_ || !output_ || !dirty_) return;
        for (uint32_t phase = 0; phase < 3; ++phase) {
            for (uint32_t tier = 0; tier < 3; ++tier) {
                const auto phase_value = static_cast<SupplyTelemetryPhase>(phase);
                const auto tier_value = static_cast<ExpertTier>(tier);
                const auto& summary = summary_for(phase_value, tier_value);
                if (summary.request_count == 0 && summary.demotion_attempts == 0 &&
                    summary.demotion_drops == 0) {
                    continue;
                }
                output_ << summary_json(phase_value, tier_value, summary) << '\n';
            }
        }
        for (const auto& event : transfer_events_) {
            output_ << event << '\n';
        }
        output_.flush();
        transfer_events_.clear();
        for (auto& phase : summaries_) {
            for (auto& summary : phase) {
                summary = Summary{};
            }
        }
        dirty_ = false;
    }

private:
    static constexpr uint32_t kPhaseCount = 3;
    static constexpr uint32_t kTierCount = 3;

    bool enabled_{false};
    std::string run_id_;
    std::ofstream output_;
    SupplyTelemetryPhase current_phase_{SupplyTelemetryPhase::Decode};
    uint64_t boundary_id_{0};
    uint64_t baseline_vm_swap_bytes_{0};
    bool dirty_{false};
    Summary summaries_[kPhaseCount][kTierCount]{};
    std::vector<std::string> transfer_events_;

    Summary& summary_for(SupplyTelemetryPhase phase, ExpertTier tier) {
        return summaries_[static_cast<uint32_t>(phase)][static_cast<uint32_t>(tier)];
    }

    const Summary& summary_for(SupplyTelemetryPhase phase, ExpertTier tier) const {
        return summaries_[static_cast<uint32_t>(phase)][static_cast<uint32_t>(tier)];
    }

    static uint64_t read_vm_swap_bytes() {
        return read_status_kib("VmSwap:") * 1024ULL;
    }

    static uint64_t read_rss_bytes() {
        return read_status_kib("VmRSS:") * 1024ULL;
    }

    static uint64_t read_status_kib(const char* field) {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind(field, 0) != 0) continue;
            std::istringstream values(line.substr(std::char_traits<char>::length(field)));
            uint64_t value = 0;
            std::string unit;
            if (values >> value >> unit) {
                return value;
            }
        }
        return 0;
    }

    void observe_process_memory(Summary& summary) {
        summary.rss_bytes_peak = std::max(summary.rss_bytes_peak, read_rss_bytes());
        const uint64_t swap_bytes = read_vm_swap_bytes();
        if (swap_bytes >= baseline_vm_swap_bytes_) {
            summary.vm_swap_bytes_delta = static_cast<int64_t>(swap_bytes - baseline_vm_swap_bytes_);
        } else {
            summary.vm_swap_bytes_delta = -static_cast<int64_t>(baseline_vm_swap_bytes_ - swap_bytes);
        }
    }

    static const char* phase_name(SupplyTelemetryPhase phase) {
        switch (phase) {
        case SupplyTelemetryPhase::Warmup: return "warmup";
        case SupplyTelemetryPhase::Prefill: return "prefill";
        case SupplyTelemetryPhase::Decode: return "decode";
        }
        return "decode";
    }

    static const char* tier_name(ExpertTier tier) {
        switch (tier) {
        case ExpertTier::HOT_VRAM: return "hot";
        case ExpertTier::WARM_HOST: return "warm";
        case ExpertTier::COLD_NVME: return "cold";
        }
        return "none";
    }

    std::string summary_json(
        SupplyTelemetryPhase phase,
        ExpertTier tier,
        const Summary& summary
    ) const {
        std::ostringstream json;
        json << "{\"record_type\":\"phase_summary\",\"schema_version\":"
             << schema_version << ",\"run_id\":\"" << run_id_
             << "\",\"boundary_id\":" << boundary_id_
             << ",\"phase\":\"" << phase_name(phase)
             << "\",\"source_tier\":\"" << tier_name(tier)
             << "\",\"request_count\":" << summary.request_count
             << ",\"decode_token_count\":" << summary.decode_token_count
             << ",\"bytes_from_nvme\":" << summary.bytes_from_nvme
             << ",\"bytes_from_host\":" << summary.bytes_from_host
             << ",\"logical_bytes_from_warm\":" << summary.logical_bytes_from_warm
             << ",\"logical_bytes_from_transient_staging\":" << summary.logical_bytes_from_transient_staging
             << ",\"h2d_bytes\":" << summary.h2d_bytes
             << ",\"d2h_bytes\":" << summary.d2h_bytes
             << ",\"nvme_read_service_ns\":" << summary.nvme_read_service_ns
             << ",\"nvme_completion_wait_ns\":" << summary.nvme_completion_wait_ns
             << ",\"h2d_enqueue_to_ready_ns\":" << summary.h2d_enqueue_to_ready_ns
             << ",\"gpu_readiness_wait_ns\":" << summary.gpu_readiness_wait_ns
             << ",\"optional_demotion_wait_ns\":" << summary.optional_demotion_wait_ns
             << ",\"staging_wait_ns\":" << summary.staging_wait_ns
             << ",\"staging_reuse_wait_ns\":" << summary.staging_reuse_wait_ns
             << ",\"demotion_attempts\":" << summary.demotion_attempts
             << ",\"demotion_completions\":" << summary.demotion_completions
             << ",\"demotion_drops\":" << summary.demotion_drops
             << ",\"demotion_drop_reasons\":{";
        bool first_reason = true;
        for (const auto& [reason, count] : summary.demotion_drop_reasons) {
            if (!first_reason) json << ',';
            first_reason = false;
            json << '"' << reason << "\":" << count;
        }
        json << "},\"hot_occupancy_min\":" << summary.hot_occupancy_min
             << ",\"hot_occupancy_max\":" << summary.hot_occupancy_max
             << ",\"warm_valid_slots_min\":" << summary.warm_valid_slots_min
             << ",\"warm_valid_slots_max\":" << summary.warm_valid_slots_max
             << ",\"pending_transfers_max\":" << summary.pending_transfers_max
             << ",\"demotion_queue_depth_max\":" << summary.demotion_queue_depth_max
             << ",\"warm_pinned_bytes\":" << summary.warm_pinned_bytes
             << ",\"warm_unpinned_bytes\":" << summary.warm_unpinned_bytes
             << ",\"rss_bytes_peak\":" << summary.rss_bytes_peak
             << ",\"vm_swap_bytes_delta\":" << summary.vm_swap_bytes_delta
             << "}";
        return json.str();
    }
};

} // namespace aeon::core