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

# --- Legacy graph tests ------------------------------------------------------
# The pre-rewrite layer-schedule tests and the circular kernel checks were
# deleted with the graph (2026-09-15). What remains are the model-backed parity
# tests: they load real checkpoint weights and compare against
# `reference/v4_int4_reference.hpp`, which is written independently of the code
# under test. They are kept gated because they exercise kernels the rewrite
# audits, so they are not evidence about the new graph — but they are the closest
# thing to a tier-0 anchor and are worth re-deriving before deleting.
if(AEON_BUILD_TESTS)
    # Real INT4 expert parity: fused kernels versus the independent reference.
    aeon_add_test(test_v4_real_expert_parity SOURCES tests/test_v4_real_expert_parity.cpp)

    # Real dense projection parity: wq_a/wq_b/wo_a/wo_b, HC, RMSNorm versus the
    # independent reference. Depends on kernels/v4_attention.hpp.
    aeon_add_test(test_v4_real_dense_parity SOURCES tests/test_v4_real_dense_parity.cpp)
endif()
