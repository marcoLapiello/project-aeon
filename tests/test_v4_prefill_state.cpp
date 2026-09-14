#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Pipeline;
using aeon::core::V4PrefillExecutionPath;
using aeon::core::V4PipelineStateSnapshot;

constexpr size_t kVocabularySize = 129280;
constexpr float kLogitTolerance = 0.02f;

struct PrefillResult {
    uint32_t next_token{0};
    uint32_t continuation_token{0};
    std::vector<float> logits;
    V4PipelineStateSnapshot state;
};

std::vector<float> snapshot_logits(V4Pipeline& pipeline) {
    std::vector<half> device_logits(kVocabularySize);
    assert(hipMemcpy(
        device_logits.data(),
        pipeline.scratch.d_logits,
        device_logits.size() * sizeof(half),
        hipMemcpyDeviceToHost) == hipSuccess);
    std::vector<float> logits(device_logits.size());
    for (size_t index = 0; index < device_logits.size(); ++index) {
        logits[index] = __half2float(device_logits[index]);
    }
    return logits;
}

void assert_state_equal(
    const V4PipelineStateSnapshot& expected,
    const V4PipelineStateSnapshot& actual
) {
    assert(expected.current_seq_len == actual.current_seq_len);
    assert(expected.layers.size() == actual.layers.size());
    for (size_t layer_index = 0; layer_index < expected.layers.size(); ++layer_index) {
        const auto& lhs = expected.layers[layer_index];
        const auto& rhs = actual.layers[layer_index];
        assert(lhs.local_valid_count == rhs.local_valid_count);
        assert(lhs.compressor_partial_count == rhs.compressor_partial_count);
        assert(lhs.compressed_entry_count == rhs.compressed_entry_count);
        assert(lhs.indexer_candidate_count == rhs.indexer_candidate_count);
        const auto compare_exact = [&](const char* name, const std::vector<uint8_t>& left,
                                       const std::vector<uint8_t>& right) {
            if (left == right) return;
            size_t first_difference = 0;
            while (first_difference < left.size() && first_difference < right.size() &&
                   left[first_difference] == right[first_difference]) {
                ++first_difference;
            }
            std::cerr << "State mismatch at layer " << layer_index << " field " << name
                      << " byte " << first_difference << " sizes " << left.size() << "/"
                      << right.size();
            if (first_difference < left.size() && first_difference < right.size()) {
                std::cerr << " values " << static_cast<unsigned>(left[first_difference]) << "/"
                          << static_cast<unsigned>(right[first_difference]);
            }
            std::cerr << '\n';
            assert(false);
        };
        const auto compare_half = [&](const char* name, const std::vector<uint8_t>& left,
                                      const std::vector<uint8_t>& right, float tolerance) {
            assert(left.size() == right.size());
            assert(left.size() % sizeof(half) == 0);
            const auto* left_values = reinterpret_cast<const half*>(left.data());
            const auto* right_values = reinterpret_cast<const half*>(right.data());
            for (size_t index = 0; index < left.size() / sizeof(half); ++index) {
                const float left_value = __half2float(left_values[index]);
                const float right_value = __half2float(right_values[index]);
                if (std::fabs(left_value - right_value) > tolerance) {
                    std::cerr << "State mismatch at layer " << layer_index << " field " << name
                              << " element " << index << " values " << left_value << "/"
                              << right_value << " tolerance " << tolerance << '\n';
                    assert(false);
                }
            }
        };
        const auto compare_float = [&](const char* name, const std::vector<uint8_t>& left,
                                       const std::vector<uint8_t>& right, float tolerance) {
            assert(left.size() == right.size());
            assert(left.size() % sizeof(float) == 0);
            const auto* left_values = reinterpret_cast<const float*>(left.data());
            const auto* right_values = reinterpret_cast<const float*>(right.data());
            for (size_t index = 0; index < left.size() / sizeof(float); ++index) {
                if (std::fabs(left_values[index] - right_values[index]) > tolerance) {
                    std::cerr << "State mismatch at layer " << layer_index << " field " << name
                              << " element " << index << " values " << left_values[index] << "/"
                              << right_values[index] << " tolerance " << tolerance << '\n';
                    assert(false);
                }
            }
        };
        compare_half("local_key_cache", lhs.local_key_cache, rhs.local_key_cache, 0.02f);
        compare_half("local_value_cache", lhs.local_value_cache, rhs.local_value_cache, 0.02f);
        compare_exact("local_positions", lhs.local_positions, rhs.local_positions);
        compare_half("compressed_key_cache", lhs.compressed_key_cache, rhs.compressed_key_cache, 0.02f);
        compare_half("compressed_value_cache", lhs.compressed_value_cache, rhs.compressed_value_cache, 0.02f);
        compare_exact("compressed_positions", lhs.compressed_positions, rhs.compressed_positions);
        compare_float("compressor_partial_kv", lhs.compressor_partial_kv, rhs.compressor_partial_kv, 0.002f);
        compare_float("compressor_partial_score", lhs.compressor_partial_score, rhs.compressor_partial_score, 0.002f);
        compare_exact("compressor_partial_positions", lhs.compressor_partial_positions,
                      rhs.compressor_partial_positions);
        compare_half("indexer_key_cache", lhs.indexer_key_cache, rhs.indexer_key_cache, 0.02f);
        compare_exact("indexer_positions", lhs.indexer_positions, rhs.indexer_positions);
        compare_float("indexer_partial_kv", lhs.indexer_partial_kv, rhs.indexer_partial_kv, 0.002f);
        compare_float("indexer_partial_score", lhs.indexer_partial_score, rhs.indexer_partial_score, 0.002f);
        compare_exact("indexer_partial_positions", lhs.indexer_partial_positions,
                      rhs.indexer_partial_positions);
        compare_half("indexer_query", lhs.indexer_query, rhs.indexer_query, 0.02f);
        compare_float("indexer_weights", lhs.indexer_weights, rhs.indexer_weights, 0.002f);
        compare_float("indexer_scores", lhs.indexer_scores, rhs.indexer_scores, 0.1f);
        compare_exact("indexer_topk_indices", lhs.indexer_topk_indices, rhs.indexer_topk_indices);
    }
}

PrefillResult run_with_boundaries(
    V4Pipeline& pipeline,
    const std::vector<uint32_t>& prompt,
    const std::vector<size_t>& boundaries
) {
    pipeline.reset_generation_state();
    size_t start = 0;
    uint32_t next_token = 0;
    for (const size_t requested_end : boundaries) {
        const size_t end = std::min(requested_end, prompt.size());
        if (end <= start) continue;
        next_token = pipeline.prefill(
            std::span<const uint32_t>(prompt.data() + start, end - start),
            static_cast<uint32_t>(start));
        start = end;
    }
    assert(start == prompt.size());
    const auto logits = snapshot_logits(pipeline);
    const auto state = pipeline.snapshot_generation_state();
    const uint32_t continuation_token = pipeline.step(
        next_token, static_cast<uint32_t>(prompt.size()), aeon::core::RoutingPhase::Decode);
    return {next_token, continuation_token, logits, state};
}

PrefillResult run_with_chunk_size(
    V4Pipeline& pipeline,
    const std::vector<uint32_t>& prompt,
    size_t chunk_size
) {
    std::vector<size_t> boundaries;
    for (size_t end = chunk_size; end < prompt.size(); end += chunk_size) {
        boundaries.push_back(end);
    }
    boundaries.push_back(prompt.size());
    return run_with_boundaries(pipeline, prompt, boundaries);
}

PrefillResult run_with_batched_request(
    V4Pipeline& pipeline,
    const std::vector<uint32_t>& prompt,
    size_t requested_batch_size
) {
    pipeline.reset_generation_state();
    const auto report = pipeline.prefill_batched(
        std::span<const uint32_t>(prompt),
        0,
        requested_batch_size);
    assert(report.token_count == prompt.size());
    assert(report.execution_path == V4PrefillExecutionPath::Batched);
    const auto logits = snapshot_logits(pipeline);
    const auto state = pipeline.snapshot_generation_state();
    const uint32_t continuation_token = pipeline.step(
        report.next_token, static_cast<uint32_t>(prompt.size()), aeon::core::RoutingPhase::Decode);
    return {report.next_token, continuation_token, logits, state};
}

void assert_reset_is_empty(V4Pipeline& pipeline) {
    pipeline.reset_generation_state();
    const auto state = pipeline.snapshot_generation_state();
    assert(state.current_seq_len == 0);
    for (const auto& layer : state.layers) {
        assert(layer.local_valid_count == 0);
        assert(layer.compressor_partial_count == 0);
        assert(layer.compressed_entry_count == 0);
        assert(layer.indexer_candidate_count == 0);
    }
}

} // namespace

int main(int argc, char** argv) {
    aeon::core::select_compute_device(true);

    constexpr size_t kPromptTokens = 132;
    constexpr size_t kDefaultWarmHostGiB = 35;
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    const size_t warm_host_gib = argc > 1
        ? std::stoull(argv[1])
        : kDefaultWarmHostGiB;
    aeon::core::AeonRuntimeConfig runtime_config;
    runtime_config.context_size = static_cast<uint32_t>(kPromptTokens + 1);
    runtime_config.warm_host_bytes = warm_host_gib * 1024ULL * 1024ULL * 1024ULL;
    runtime_config.deterministic_expert_accumulation = true;

    std::cout << "V4 Stage 5 resource profile: Warm host budget="
              << warm_host_gib << " GiB (pass 0 for the cold-only control)" << std::endl;

    V4Pipeline pipeline;
    pipeline.initialize(model_dir, runtime_config);

    std::vector<uint32_t> prompt;
    prompt.reserve(kPromptTokens);
    for (size_t position = 0; position < kPromptTokens; ++position) {
        prompt.push_back(1u + static_cast<uint32_t>(position));
    }

    bool rejected_gap = false;
    try {
        pipeline.prefill(std::span<const uint32_t>(prompt.data(), 1), 1);
    } catch (const std::invalid_argument&) {
        rejected_gap = true;
    }
    assert(rejected_gap);

    pipeline.reset_generation_state();
    const auto required_batch = pipeline.prefill_batched(
        std::span<const uint32_t>(prompt),
        0,
        16,
        false);
    assert(required_batch.token_count == prompt.size());
    assert(required_batch.execution_path == V4PrefillExecutionPath::Batched);
    pipeline.reset_generation_state();

    assert_reset_is_empty(pipeline);
    const auto one_shot = run_with_boundaries(pipeline, prompt, {prompt.size()});
    const auto one_shot_repeat = run_with_boundaries(pipeline, prompt, {prompt.size()});
    const auto serialized = run_with_chunk_size(pipeline, prompt, 1);
    const auto aligned_four = run_with_chunk_size(pipeline, prompt, 4);
    const auto aligned_128 = run_with_chunk_size(pipeline, prompt, 128);
    const auto requested_batch = run_with_batched_request(pipeline, prompt, 16);
    const auto boundary_splits = run_with_boundaries(
        pipeline, prompt, {1, 3, 4, 7, 16, 127, 128, prompt.size()});

    const std::vector<std::pair<const char*, const PrefillResult*>> comparisons = {
        {"one-shot-repeat", &one_shot_repeat},
        {"serialized", &serialized},
        {"aligned-four", &aligned_four},
        {"aligned-128", &aligned_128},
        {"requested-batch", &requested_batch},
        {"boundary-splits", &boundary_splits},
    };
    for (const auto& [name, result] : comparisons) {
        std::cerr << "Comparing one-shot with " << name << '\n';
        float max_logit_difference = 0.0f;
        size_t max_logit_difference_index = 0;
        for (size_t index = 0; index < one_shot.logits.size(); ++index) {
            const float difference = std::fabs(one_shot.logits[index] - result->logits[index]);
            if (difference > max_logit_difference) {
                max_logit_difference = difference;
                max_logit_difference_index = index;
            }
        }
        if (max_logit_difference > kLogitTolerance) {
            std::cerr << "Logit mismatch: max difference=" << max_logit_difference
                      << " at token " << max_logit_difference_index
                      << " values " << one_shot.logits[max_logit_difference_index] << "/"
                      << result->logits[max_logit_difference_index]
                      << " tolerance " << kLogitTolerance << '\n';
            assert(false);
        }
        if (result->next_token != one_shot.next_token) {
            std::cerr << "First generated token mismatch: one-shot=" << one_shot.next_token
                      << " " << name << "=" << result->next_token
                      << '\n';
        }
        assert_state_equal(one_shot.state, result->state);
        if (result->next_token != one_shot.next_token) {
            assert(false);
        }
        if (result->continuation_token != one_shot.continuation_token) {
            std::cerr << "Continuation token mismatch: one-shot=" << one_shot.continuation_token
                      << " " << name << "=" << result->continuation_token << '\n';
        }
        assert(result->continuation_token == one_shot.continuation_token);
    }

    std::cout << "V4 Stage 5 prefill passed: one-shot, serialized, aligned, and boundary-split "
              << "state/next-token equivalence through position " << (kPromptTokens - 1) << std::endl;
    return 0;
}