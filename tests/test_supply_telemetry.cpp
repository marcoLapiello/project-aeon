#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const std::string path = "/tmp/aeon_supply_telemetry_test.jsonl";
    aeon::core::SupplyTelemetry telemetry;
    telemetry.enable_jsonl(path, "telemetry-test");
    telemetry.set_phase(aeon::core::RoutingPhase::Decode);
    telemetry.record_request(
        aeon::core::SupplyTelemetryPhase::Decode,
        aeon::core::ExpertTier::COLD_NVME,
        aeon::core::AEON_EXPERT_BYTES,
        aeon::core::AEON_EXPERT_BYTES,
        aeon::core::AEON_EXPERT_BYTES,
        0,
        aeon::core::AEON_EXPERT_BYTES
    );
    telemetry.record_demotion_attempt(aeon::core::ExpertTier::HOT_VRAM);
    telemetry.record_demotion_drop(aeon::core::ExpertTier::HOT_VRAM, "queue_pressure");
    telemetry.observe_occupancy(
        2, 1, 1, 1, aeon::core::AEON_EXPERT_BYTES, 0,
        aeon::core::ExpertTier::COLD_NVME);
    telemetry.record_transfer_event(7, 3, "d2h", 1, 0, "drop", "queue_pressure");
    telemetry.flush();
    telemetry.flush();
    telemetry.disable();

    std::ifstream input(path);
    assert(input.good());
    std::string line;
    uint32_t line_count = 0;
    bool found_summary = false;
    bool found_transfer = false;
    bool found_demotion_attempt = false;
    bool found_queue_pressure_reason = false;
    while (std::getline(input, line)) {
        ++line_count;
        found_summary = found_summary || line.find("phase_summary") != std::string::npos;
        found_transfer = found_transfer || line.find("transfer_event") != std::string::npos;
        assert(line.find("\"schema_version\":1") != std::string::npos);
        assert(line.find("\"optional_demotion_wait_ns\":0") != std::string::npos ||
               line.find("transfer_event") != std::string::npos);
        if (line.find("phase_summary") != std::string::npos) {
            found_demotion_attempt = found_demotion_attempt ||
                line.find("\"demotion_attempts\":1") != std::string::npos;
            found_queue_pressure_reason = found_queue_pressure_reason ||
                line.find("\"queue_pressure\":1") != std::string::npos;
            assert(line.find("\"rss_bytes_peak\":") != std::string::npos);
            assert(line.find("\"vm_swap_bytes_delta\":") != std::string::npos);
        }
    }
    assert(line_count == 3);
    assert(found_summary);
    assert(found_transfer);
    assert(found_demotion_attempt);
    assert(found_queue_pressure_reason);

    std::cout << "Supply telemetry JSONL contract passed\n";
    return 0;
}
