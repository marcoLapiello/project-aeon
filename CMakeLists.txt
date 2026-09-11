cmake_minimum_required(VERSION 3.28)
project(aeon VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ROCm installation path discovery
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    else()
        message(FATAL_ERROR "ROCm not found. Please set ROCM_PATH.")
    endif()
endif()
message(STATUS "Project Aeon: Using ROCm path: ${ROCM_PATH}")

# Target AMD GPU Architecture (Navi 31 / RX 7900 series)
set(AEON_GPU_TARGET "gfx1100" CACHE STRING "Target AMD GPU architecture")
message(STATUS "Project Aeon: Offload target architecture: ${AEON_GPU_TARGET}")

# Compiler Flags
# -mno-wavefrontsize64 enforces Wave32 execution mode on RDNA3
add_compile_options(
    -O3
    -Wall
    -Wextra
    -Wno-unused-parameter
    --offload-arch=${AEON_GPU_TARGET}
    -mno-wavefrontsize64
)

# Global include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ROCM_PATH}/include
)

# Global link directories
link_directories(
    ${ROCM_PATH}/lib
)

# Output binaries to bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Smoke test executable to verify compiler and toolchain sanity
add_executable(smoke_check src/smoke.cpp)
target_link_libraries(smoke_check PRIVATE amdhip64)

# Hardware inspection & topology discovery tool
add_executable(aeon_info tools/aeon_info.cpp)
target_link_libraries(aeon_info PRIVATE amdhip64)

# Full-model routing profile driver with durable incremental aggregation
add_executable(profile_routing tools/profile_routing.cpp)
target_link_libraries(profile_routing PRIVATE amdhip64)
execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short HEAD
    OUTPUT_VARIABLE AEON_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT AEON_GIT_COMMIT)
    set(AEON_GIT_COMMIT "unknown")
endif()
target_compile_definitions(profile_routing PRIVATE AEON_GIT_COMMIT=\"${AEON_GIT_COMMIT}\")

# Native text-in/text-out CLI for the verified DSV4 formatter and pipeline
add_executable(aeon_chat
    tools/aeon_chat.cpp
    src/text/dsv4_tokenizer.cpp
    src/text/dsv4_chat_formatter.cpp
    src/text/text_generation.cpp
)
target_link_libraries(aeon_chat PRIVATE amdhip64)

# Hardware validation: single-tile Wave32 WMMA smoke test
add_executable(test_wmma_tile tests/test_wmma_tile.cpp)
target_link_libraries(test_wmma_tile PRIVATE amdhip64)

# Hardware benchmark: tiled Wave32 WMMA GEMM
add_executable(bench_wmma_gemm tests/bench_wmma_gemm.cpp)
target_link_libraries(bench_wmma_gemm PRIVATE amdhip64)

# I/O validation: 4KB sector-aligned direct reader
add_executable(test_direct_io tests/test_direct_io.cpp)
target_link_libraries(test_direct_io PRIVATE amdhip64)

# Phase 2 Spike 3 validation: model-backed batched O_DIRECT expert reads
add_executable(test_model_direct_io tests/test_model_direct_io.cpp)
target_link_libraries(test_model_direct_io PRIVATE amdhip64)
add_test(NAME test_model_direct_io COMMAND test_model_direct_io)
set_tests_properties(test_model_direct_io PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Phase 2 Spike 3 investigation: request shape and queue-width benchmark
add_executable(bench_model_direct_io tests/bench_model_direct_io.cpp)
target_link_libraries(bench_model_direct_io PRIVATE amdhip64)

# Phase 2 Spike 3 investigation: native pipeline direct-I/O versus mmap A/B
add_executable(bench_async_prefetch_io_modes tests/bench_async_prefetch_io_modes.cpp)
target_link_libraries(bench_async_prefetch_io_modes PRIVATE amdhip64)

# Transfer benchmark: concurrent compute and SDMA overlap
add_executable(bench_async_overlap tests/bench_async_overlap.cpp)
target_link_libraries(bench_async_overlap PRIVATE amdhip64)

# Legacy diagnostic: single-layer toy MoE cache pipeline
add_executable(test_toy_moe_layer tests/test_toy_moe_layer.cpp)
target_link_libraries(test_toy_moe_layer PRIVATE amdhip64)

# Regression test: DeepSeek-V4 configuration parser
add_executable(test_config_parser tests/test_config_parser.cpp)
target_link_libraries(test_config_parser PRIVATE amdhip64)

# Regression test: Safetensors header parser
add_executable(test_safetensors_parser tests/test_safetensors_parser.cpp)
target_link_libraries(test_safetensors_parser PRIVATE amdhip64)

# Regression test: Wave32 RMSNorm and clamped SwiGLU
add_executable(test_swiglu_clamp tests/test_swiglu_clamp.cpp)
target_link_libraries(test_swiglu_clamp PRIVATE amdhip64)

# Regression test: Hyper-Connections Sinkhorn and expansion
add_executable(test_hc_sinkhorn tests/test_hc_sinkhorn.cpp)
target_link_libraries(test_hc_sinkhorn PRIVATE amdhip64)
add_test(NAME test_hc_sinkhorn COMMAND test_hc_sinkhorn)

# Regression test: fused INT4-to-FP16 Wave32 WMMA
add_executable(test_w4a16_wmma tests/test_w4a16_wmma.cpp)
target_link_libraries(test_w4a16_wmma PRIVATE amdhip64)

# Host-side validation: parallel W4A16 swizzle layout contract
add_executable(test_w4a16_swizzle tests/test_w4a16_swizzle.cpp)
target_link_libraries(test_w4a16_swizzle PRIVATE amdhip64)
add_test(NAME test_w4a16_swizzle COMMAND test_w4a16_swizzle)
set_tests_properties(test_w4a16_swizzle PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Silicon validation: parallel swizzled W4A16 decode GEMV
add_executable(test_w4a16_swizzled_gemv tests/test_w4a16_swizzled_gemv.cpp)
target_link_libraries(test_w4a16_swizzled_gemv PRIVATE amdhip64)
add_test(NAME test_w4a16_swizzled_gemv COMMAND test_w4a16_swizzled_gemv)
set_tests_properties(test_w4a16_swizzled_gemv PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Silicon validation: fused parallel swizzled W1/W3 GEMV
add_executable(test_w4a16_swizzled_dual_gemv tests/test_w4a16_swizzled_dual_gemv.cpp)
target_link_libraries(test_w4a16_swizzled_dual_gemv PRIVATE amdhip64)
add_test(NAME test_w4a16_swizzled_dual_gemv COMMAND test_w4a16_swizzled_dual_gemv)
set_tests_properties(test_w4a16_swizzled_dual_gemv PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Silicon validation: six-expert fused W1/W3 plus clamped SwiGLU
add_executable(test_aeon_moe_fused_w13 tests/test_aeon_moe_fused_w13.cpp)
target_link_libraries(test_aeon_moe_fused_w13 PRIVATE amdhip64)
add_test(NAME test_aeon_moe_fused_w13 COMMAND test_aeon_moe_fused_w13)
set_tests_properties(test_aeon_moe_fused_w13 PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Silicon validation: six-expert fused W2 weighted accumulation
add_executable(test_aeon_moe_fused_w2 tests/test_aeon_moe_fused_w2.cpp)
target_link_libraries(test_aeon_moe_fused_w2 PRIVATE amdhip64)
add_test(NAME test_aeon_moe_fused_w2 COMMAND test_aeon_moe_fused_w2)
set_tests_properties(test_aeon_moe_fused_w2 PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark: current versus parallel swizzled W4A16 decode GEMV
add_executable(bench_w4a16_swizzled_gemv tests/bench_w4a16_swizzled_gemv.cpp)
target_link_libraries(bench_w4a16_swizzled_gemv PRIVATE amdhip64)

# Benchmark: two current W1/W3 launches versus one parallel dual launch
add_executable(bench_w4a16_swizzled_dual_gemv tests/bench_w4a16_swizzled_dual_gemv.cpp)
target_link_libraries(bench_w4a16_swizzled_dual_gemv PRIVATE amdhip64)

# Benchmark: six separate routed W1/W3/SwiGLU paths versus one fused dispatch
add_executable(bench_aeon_moe_fused_w13 tests/bench_aeon_moe_fused_w13.cpp)
target_link_libraries(bench_aeon_moe_fused_w13 PRIVATE amdhip64)

# Benchmark: six routed W2/accumulation paths versus one fused dispatch
add_executable(bench_aeon_moe_fused_w2 tests/bench_aeon_moe_fused_w2.cpp)
target_link_libraries(bench_aeon_moe_fused_w2 PRIVATE amdhip64)

# Regression test: dual-mode MoE router
add_executable(test_moe_router tests/test_moe_router.cpp)
target_link_libraries(test_moe_router PRIVATE amdhip64)

# Regression test: DeepSeek-V4 MoE layer forward pass
add_executable(test_v4_moe_layer tests/test_v4_moe_layer.cpp)
target_link_libraries(test_v4_moe_layer PRIVATE amdhip64)

# Regression test: sliding-window attention and rotary embeddings
add_executable(test_v4_attention tests/test_v4_attention.cpp)
target_link_libraries(test_v4_attention PRIVATE amdhip64)

# Regression test: complete DeepSeek-V4 transformer block
add_executable(test_v4_block tests/test_v4_block.cpp)
target_link_libraries(test_v4_block PRIVATE amdhip64)

# Regression test: multi-layer pipeline and KV cache
add_executable(test_v4_pipeline tests/test_v4_pipeline.cpp)
target_link_libraries(test_v4_pipeline PRIVATE amdhip64)
add_test(NAME test_v4_pipeline COMMAND test_v4_pipeline)
set_tests_properties(test_v4_pipeline PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Baseline benchmark: Safetensors autoregressive generation
add_executable(bench_v4_generation tests/bench_v4_generation.cpp)
target_link_libraries(bench_v4_generation PRIVATE amdhip64)

# Current format validation: native Aeon model loader
add_executable(test_aeon_loader tests/test_aeon_loader.cpp)
target_link_libraries(test_aeon_loader PRIVATE amdhip64)

# Parallel format validation: version-2 swizzled Aeon model loader
add_executable(test_aeon_swizzled_loader tests/test_aeon_swizzled_loader.cpp)
target_link_libraries(test_aeon_swizzled_loader PRIVATE amdhip64)

# Silicon smoke test: opt-in swizzled fused-expert pipeline path
add_executable(test_aeon_swizzled_pipeline tests/test_aeon_swizzled_pipeline.cpp)
target_link_libraries(test_aeon_swizzled_pipeline PRIVATE amdhip64)

# CPU-side validation: native DSV4 tokenizer artifact and ByteLevel-BPE parity
add_executable(test_dsv4_tokenizer tests/test_dsv4_tokenizer.cpp src/text/dsv4_tokenizer.cpp)
target_link_libraries(test_dsv4_tokenizer PRIVATE amdhip64)
add_test(NAME test_dsv4_tokenizer COMMAND test_dsv4_tokenizer)
set_tests_properties(test_dsv4_tokenizer PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# CPU-side validation: native DSV4 chat/thinking prompt formatting
add_executable(test_dsv4_chat_formatter
    tests/test_dsv4_chat_formatter.cpp
    src/text/dsv4_tokenizer.cpp
    src/text/dsv4_chat_formatter.cpp
)
target_link_libraries(test_dsv4_chat_formatter PRIVATE amdhip64)
add_test(NAME test_dsv4_chat_formatter COMMAND test_dsv4_chat_formatter)
set_tests_properties(test_dsv4_chat_formatter PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# CPU-side validation: EOS-aware token generation loop and stop reasons
add_executable(test_text_generation tests/test_text_generation.cpp src/text/text_generation.cpp)
target_link_libraries(test_text_generation PRIVATE amdhip64)
add_test(NAME test_text_generation COMMAND test_text_generation)
set_tests_properties(test_text_generation PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Current format validation: end-to-end native .aeon pipeline
add_executable(test_aeon_pipeline tests/test_aeon_pipeline.cpp)
target_link_libraries(test_aeon_pipeline PRIVATE amdhip64)
add_test(NAME test_aeon_pipeline COMMAND test_aeon_pipeline)
set_tests_properties(test_aeon_pipeline PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Measurement plumbing: optional per-layer expert selection counter
add_executable(test_routing_counter tests/test_routing_counter.cpp)
target_link_libraries(test_routing_counter PRIVATE amdhip64)
add_test(NAME test_routing_counter COMMAND test_routing_counter)
set_tests_properties(test_routing_counter PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Host-side routing profile parser and checkpoint validation
add_executable(test_routing_profile tests/test_routing_profile.cpp)
target_link_libraries(test_routing_profile PRIVATE amdhip64)
add_test(NAME test_routing_profile COMMAND test_routing_profile)
set_tests_properties(test_routing_profile PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Current architecture validation: dynamic memory budget and global pool
add_executable(test_dynamic_expert_pool tests/test_dynamic_expert_pool.cpp)
target_link_libraries(test_dynamic_expert_pool PRIVATE amdhip64)
add_test(NAME test_dynamic_expert_pool COMMAND test_dynamic_expert_pool)
set_tests_properties(test_dynamic_expert_pool PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Host-side validation: persistent Warm ownership, pending transfers, and leases
add_executable(test_expert_registry_warm_state tests/test_expert_registry_warm_state.cpp)
target_link_libraries(test_expert_registry_warm_state PRIVATE amdhip64)
add_test(NAME test_expert_registry_warm_state COMMAND test_expert_registry_warm_state)
set_tests_properties(test_expert_registry_warm_state PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Host-side validation: versioned, phase-labelled supply telemetry JSONL
add_executable(test_supply_telemetry tests/test_supply_telemetry.cpp)
target_link_libraries(test_supply_telemetry PRIVATE amdhip64)
add_test(NAME test_supply_telemetry COMMAND test_supply_telemetry)
set_tests_properties(test_supply_telemetry PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Phase 2 end-to-end smoke test: direct hot/warm initialization and cold streaming
add_executable(test_hot_warm_cold_pipeline tests/test_hot_warm_cold_pipeline.cpp)
target_link_libraries(test_hot_warm_cold_pipeline PRIVATE amdhip64)
add_test(NAME test_hot_warm_cold_pipeline COMMAND test_hot_warm_cold_pipeline)
set_tests_properties(test_hot_warm_cold_pipeline PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Phase 2 Spike 2 validation: dual-stream asynchronous SDMA prefetching
add_executable(test_async_prefetch tests/test_async_prefetch.cpp)
target_link_libraries(test_async_prefetch PRIVATE amdhip64)
add_test(NAME test_async_prefetch COMMAND test_async_prefetch)
set_tests_properties(test_async_prefetch PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Current architecture benchmark: global VRAM expert pool generation
add_executable(bench_dynamic_pool tests/bench_dynamic_pool.cpp)
target_link_libraries(bench_dynamic_pool PRIVATE amdhip64)

# Current architecture benchmark: full 43-layer autoregressive generation
add_executable(bench_full_model tests/bench_full_model.cpp)
target_link_libraries(bench_full_model PRIVATE amdhip64)







