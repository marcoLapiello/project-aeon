# Project Aeon — pre-rewrite DeepSeek-V4 graph (legacy).
#
# Everything here includes `architecture/deepseek_v4/{core,kernels,reference}/**`
# and therefore encodes the graph logic that the rewrite replaces. It is gated
# behind AEON_ENABLE_LEGACY_V4_GRAPH (default OFF) for two reasons:
#
#   1. These tests assert the old layered design and its known-wrong assumptions.
#      A green run here says nothing about the rewrite and can mask a regression
#      if it is mistaken for coverage.
#   2. Several of them are circular — they compare a kernel against an oracle
#      that shares the same helper, so they can pass on wrong logic (see the
#      anti-circularity rule in the inference pipeline plan, Part V).
#
# The set is retained only so a specific value can be re-derived from the old
# path during the rewrite. It is expected to be deleted, in full, once the new
# graph has its own end-to-end coverage. Turning it on is a deliberate act.
#
# NOTE: `test_swiglu_clamp` is deliberately NOT here — it validates the platform
# clamp kernel and only pulls `core/config.hpp` for a parameter struct.

if(NOT AEON_ENABLE_LEGACY_V4_GRAPH)
    message(STATUS "Project Aeon: legacy V4 graph DISABLED (AEON_ENABLE_LEGACY_V4_GRAPH=OFF)")
    return()
endif()

message(STATUS "Project Aeon: legacy V4 graph ENABLED — these targets test the pre-rewrite graph")

# --- Legacy tools ------------------------------------------------------------
# Native text-in/text-out CLI driven by the old pipeline.
aeon_add_executable(aeon_chat
    SOURCES
        tools/aeon_chat.cpp
        src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
        src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp
        src/infrastructure/text/text_generation.cpp)

# Stage 0 evidence: model contract and dense tensor inventory.
aeon_add_executable(aeon_model_contract SOURCES tools/aeon_model_contract.cpp)

# Full-model routing profile driver.
aeon_add_executable(profile_routing
    SOURCES tools/profile_routing.cpp
    DEFINES AEON_GIT_COMMIT="${AEON_GIT_COMMIT}")

# Opt-in full-model GPU evidence recorder (never a default CTest).
aeon_add_executable(record_v4_gpu_evidence SOURCES tools/record_v4_gpu_evidence.cpp)

# --- Legacy benchmarks -------------------------------------------------------
aeon_add_benchmark(bench_full_model SOURCES tests/bench_full_model.cpp)

# Six separate routed W1/W3/SwiGLU paths versus one fused dispatch.
aeon_add_benchmark(bench_aeon_moe_fused_w13 SOURCES tests/bench_aeon_moe_fused_w13.cpp)

# --- Legacy graph tests ------------------------------------------------------
# NOTE: test_aeon_moe_fused_w13 and test_v4_real_attention_oracle compare a
# kernel against an oracle derived from the same helper. They are circular and
# are scheduled for deletion rather than migration.
if(AEON_BUILD_TESTS)
    # MoE router (bias placement, hash path, flat top-6).
    aeon_add_test(test_moe_router SOURCES tests/test_moe_router.cpp)

    # Hyper-Connections Sinkhorn and expansion.
    aeon_add_test(test_hc_sinkhorn SOURCES tests/test_hc_sinkhorn.cpp)

    # Six-expert fused W1/W3 plus clamped SwiGLU (compares against v4_pipeline_ops).
    aeon_add_test(test_aeon_moe_fused_w13 SOURCES tests/test_aeon_moe_fused_w13.cpp)

    # Model contract and layer schedule.
    aeon_add_test(test_v4_model_contract SOURCES tests/test_v4_model_contract.cpp)

    # Sliding-window attention and rotary embeddings.
    aeon_add_test(test_v4_attention SOURCES tests/test_v4_attention.cpp)

    # Host oracle: sliding, ratio-4 CSA, ratio-128 HCA, state transitions.
    aeon_add_test(test_v4_attention_oracle SOURCES tests/test_v4_attention_oracle.cpp)

    # Model-backed host oracle: real layer 0/2/3 dense projections and traces.
    aeon_add_test(test_v4_real_attention_oracle
        SOURCES tests/test_v4_real_attention_oracle.cpp
        TIMEOUT 120)

    # Class-aware persistent attention state sizing (host).
    aeon_add_test(test_v4_layer_state SOURCES tests/test_v4_layer_state.cpp)

    # Persistent class-aware state allocation and reset (silicon).
    aeon_add_test(test_v4_layer_state_device SOURCES tests/test_v4_layer_state_device.cpp)

    # Serial compressor, indexer, and mixed attention primitives (silicon).
    aeon_add_test(test_v4_class_attention_device SOURCES tests/test_v4_class_attention_device.cpp)

    # Serial C4A/C128A boundary dispatch.
    aeon_add_test(test_v4_stage4_dispatch SOURCES tests/test_v4_stage4_dispatch.cpp)

    # HIP attention traces versus the CPU oracle.
    aeon_add_test(test_v4_stage4_trace
        SOURCES tests/test_v4_stage4_trace.cpp
        TIMEOUT 300)

    # Serialized and chunked prefill state equivalence.
    aeon_add_test(test_v4_prefill_state
        SOURCES tests/test_v4_prefill_state.cpp
        TIMEOUT 900)

    # Real INT4 expert parity (Stage 1).
    aeon_add_test(test_v4_real_expert_parity SOURCES tests/test_v4_real_expert_parity.cpp)

    # Real dense projection parity (Stage 1).
    aeon_add_test(test_v4_real_dense_parity SOURCES tests/test_v4_real_dense_parity.cpp)

    # Dynamic memory budget and global hot pool (pulls core/v4_pipeline.hpp).
    aeon_add_test(test_dynamic_expert_pool SOURCES tests/test_dynamic_expert_pool.cpp)

    # End-to-end hot/warm initialization and cold streaming (drives the pipeline).
    aeon_add_test(test_hot_warm_cold_pipeline SOURCES tests/test_hot_warm_cold_pipeline.cpp)
endif()
