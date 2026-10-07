// -----------------------------------------------------------------------------
// Gate — the parallel topology: device mapping, layer split, and refusals.
//
// Pure and CPU-only: the topology takes the visible-device count as an argument
// rather than querying HIP, so every acceptance and every refusal is exercised
// without a GPU. The gate pins the stage-major device mapping, the contiguous
// layer ranges (the 43-layer splits for pp in {1,2,3,4}, remainder to the earliest
// stages), the empty-id default, and the `"0,1,3"` parser.
// -----------------------------------------------------------------------------

#include "infrastructure/parallel/parallel_topology.hpp"
#include "tools/engine_cli.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using aeon::core::LayerRange;
using aeon::core::ParallelTopology;
using aeon::core::ParallelTopologyConfig;
using aeon::core::parse_device_ids;

namespace {

uint32_t checks = 0;
uint32_t failures = 0;

void check(const char* label, bool ok, const std::string& detail = "") {
    std::printf("  %-58s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++checks;
    if (!ok) ++failures;
}

// Resolve expecting a refusal; returns the caught message, or "" if none.
std::string expect_refusal(const ParallelTopologyConfig& config, int visible,
                           uint32_t max_tp, uint32_t layers) {
    try {
        (void)ParallelTopology::resolve(config, visible, max_tp, layers);
        return std::string();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what());
    }
}

bool ranges_equal(const std::vector<LayerRange>& got,
                  const std::vector<LayerRange>& expected) {
    if (got.size() != expected.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i].first != expected[i].first || got[i].count != expected[i].count) return false;
    }
    return true;
}

std::vector<LayerRange> stage_ranges(const ParallelTopology& topo) {
    std::vector<LayerRange> ranges;
    for (uint32_t stage = 0; stage < topo.pp(); ++stage) ranges.push_back(topo.stage_layers(stage));
    return ranges;
}

// Drive the shared engine CLI parser over a synthetic argv and return the result.
// Throws propagate (the parse cases below catch them).
aeon::tools::EngineCli cli_parse(const std::vector<std::string>& args) {
    aeon::tools::EngineCli cli;
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("aeon"));
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    const int argc = static_cast<int>(argv.size());
    for (int index = 1; index < argc; ++index) {
        (void)aeon::tools::parse_engine_flag(cli, argc, argv.data(), index);
    }
    return cli;
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  the parallel topology: device mapping, layer split, and refusals\n");
    std::printf("================================================================================\n");

    // --- the degenerate topology --------------------------------------------
    std::printf("\n[degenerate]\n");
    {
        ParallelTopologyConfig cfg;
        const auto topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("default tp=pp=1 is single device", topo.is_single_device());
        check("default device is 0", topo.device(0, 0) == 0, "device=0");
        check("single stage has all layers", topo.stage_layers(0).first == 0 &&
                                                 topo.stage_layers(0).count == 43);
        check("stage_of every layer is 0",
              topo.stage_of(0) == 0 && topo.stage_of(42) == 0);
    }

    // --- the empty-id default -----------------------------------------------
    std::printf("\n[empty-id default]\n");
    {
        ParallelTopologyConfig cfg;
        cfg.pipeline_parallel = 4;
        const auto topo = ParallelTopology::resolve(cfg, 8, 1, 43);
        bool ids_ok = topo.device_ids().size() == 4;
        for (size_t i = 0; ids_ok && i < 4; ++i) ids_ok = topo.device_ids()[i] == static_cast<int>(i);
        check("absent ids default to 0..tp*pp-1", ids_ok);
    }

    // --- the stage-major mapping --------------------------------------------
    std::printf("\n[stage-major mapping]\n");
    {
        ParallelTopologyConfig cfg;
        cfg.device_ids = {3, 1, 0, 2};
        cfg.tensor_parallel = 2;
        cfg.pipeline_parallel = 2;
        const auto topo = ParallelTopology::resolve(cfg, 4, 8, 43);
        check("device(0,0)", topo.device(0, 0) == 3, "3");
        check("device(0,1)", topo.device(0, 1) == 1, "1");
        check("device(1,0)", topo.device(1, 0) == 0, "0");
        check("device(1,1)", topo.device(1, 1) == 2, "2");
    }

    // --- the 43-layer splits -------------------------------------------------
    std::printf("\n[layer splits over 43 layers]\n");
    {
        ParallelTopologyConfig cfg;
        auto topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("pp=1 -> [0,43)", ranges_equal(stage_ranges(topo), {{0, 43}}));

        cfg.pipeline_parallel = 2;
        topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("pp=2 -> 22/21", ranges_equal(stage_ranges(topo), {{0, 22}, {22, 21}}));

        cfg.pipeline_parallel = 3;
        topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("pp=3 -> 15/14/14",
              ranges_equal(stage_ranges(topo), {{0, 15}, {15, 14}, {29, 14}}));

        cfg.pipeline_parallel = 4;
        topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("pp=4 -> 11/11/11/10",
              ranges_equal(stage_ranges(topo), {{0, 11}, {11, 11}, {22, 11}, {33, 10}}));
    }

    // --- stage_of is total across boundaries ---------------------------------
    std::printf("\n[stage_of]\n");
    {
        ParallelTopologyConfig cfg;
        cfg.pipeline_parallel = 3;
        const auto topo = ParallelTopology::resolve(cfg, 4, 1, 43);
        check("layer 0 -> stage 0", topo.stage_of(0) == 0);
        check("layer 14 -> stage 0", topo.stage_of(14) == 0);
        check("layer 15 -> stage 1", topo.stage_of(15) == 1);
        check("layer 28 -> stage 1", topo.stage_of(28) == 1);
        check("layer 29 -> stage 2", topo.stage_of(29) == 2);
        check("layer 42 -> stage 2", topo.stage_of(42) == 2);
    }

    // --- the refusals --------------------------------------------------------
    std::printf("\n[refusals]\n");
    {
        ParallelTopologyConfig cfg;
        cfg.device_ids = {0, 0};
        cfg.tensor_parallel = 2;
        check("duplicate id refused", !expect_refusal(cfg, 4, 8, 43).empty(),
              expect_refusal(cfg, 4, 8, 43));

        cfg.device_ids = {0, 5};
        cfg.tensor_parallel = 2;
        cfg.pipeline_parallel = 1;
        check("out-of-range id refused", !expect_refusal(cfg, 4, 8, 43).empty(),
              expect_refusal(cfg, 4, 8, 43));

        cfg.device_ids = {0, 1, 2};
        cfg.tensor_parallel = 2;
        cfg.pipeline_parallel = 2;
        check("ids != tp*pp refused", !expect_refusal(cfg, 4, 8, 43).empty(),
              expect_refusal(cfg, 4, 8, 43));

        cfg.device_ids = {};
        cfg.tensor_parallel = 3;
        cfg.pipeline_parallel = 1;
        check("tp does not divide max_tp refused", !expect_refusal(cfg, 8, 8, 43).empty(),
              expect_refusal(cfg, 8, 8, 43));

        cfg.tensor_parallel = 8;
        cfg.pipeline_parallel = 1;
        check("tp dividing max_tp accepted", expect_refusal(cfg, 8, 8, 43).empty());

        cfg.tensor_parallel = 1;
        cfg.pipeline_parallel = 44;
        check("pp > num_layers refused", !expect_refusal(cfg, 44, 1, 43).empty(),
              expect_refusal(cfg, 44, 1, 43));

        cfg.tensor_parallel = 0;
        cfg.pipeline_parallel = 1;
        check("zero tp refused", !expect_refusal(cfg, 4, 8, 43).empty());

        cfg.tensor_parallel = 1;
        cfg.pipeline_parallel = 0;
        check("zero pp refused", !expect_refusal(cfg, 4, 8, 43).empty());

        cfg.tensor_parallel = 1;
        cfg.pipeline_parallel = 1;
        check("zero max_tp refused", !expect_refusal(cfg, 4, 0, 43).empty());
        check("zero layers refused", !expect_refusal(cfg, 4, 1, 0).empty());
    }

    // --- the parser ----------------------------------------------------------
    std::printf("\n[parse_device_ids]\n");
    {
        check("empty -> empty", parse_device_ids("").empty());
        check("\"0\" -> {0}", parse_device_ids("0") == std::vector<int>{0});
        check("\"0,1,3\"", parse_device_ids("0,1,3") == std::vector<int>({0, 1, 3}));
        check("whitespace trimmed", parse_device_ids(" 0 , 1 ,2 ") == std::vector<int>({0, 1, 2}));

        bool threw = false;
        try { (void)parse_device_ids("0,,1"); } catch (const std::invalid_argument&) { threw = true; }
        check("double comma refused", threw);

        threw = false;
        try { (void)parse_device_ids("0,x"); } catch (const std::invalid_argument&) { threw = true; }
        check("non-numeric refused", threw);

        threw = false;
        try { (void)parse_device_ids("1,-2"); } catch (const std::invalid_argument&) { threw = true; }
        check("negative refused", threw);

        threw = false;
        try { (void)parse_device_ids("1,2,"); } catch (const std::invalid_argument&) { threw = true; }
        check("trailing comma refused", threw);
    }

    // --- the shared engine CLI flags ----------------------------------------
    std::printf("\n[engine_cli parallel flags]\n");
    {
        const auto cli = cli_parse({"--device-ids", "0,1,3", "--tensor-parallel", "2",
                                    "--pipeline-parallel", "4",
                                    "--gpu-memory-utilization", "0.9"});
        check("--device-ids", cli.device_ids == std::vector<int>({0, 1, 3}));
        check("--tensor-parallel", cli.tensor_parallel == 2);
        check("--pipeline-parallel", cli.pipeline_parallel == 4);
        check("--gpu-memory-utilization in range",
              cli.gpu_memory_utilization > 0.89 && cli.gpu_memory_utilization < 0.91);

        check("--gpu-memory-utilization 0.99 accepted",
              cli_parse({"--gpu-memory-utilization", "0.99"}).gpu_memory_utilization == 0.99);

        for (const char* bad : {"0", "1.0", "-0.5", "abc", "0.95x"}) {
            bool rejected = false;
            try {
                (void)cli_parse({"--gpu-memory-utilization", bad});
            } catch (const std::exception&) {
                rejected = true;
            }
            check((std::string("--gpu-memory-utilization ") + bad + " refused").c_str(), rejected);
        }
    }

    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  %u checks, %u failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
