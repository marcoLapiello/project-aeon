cmake_minimum_required(VERSION 3.28)
project(aeon VERSION 0.1.0 LANGUAGES CXX)

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

# Phase 0 Spike 2.1: Single-tile Wave32 WMMA smoke test
add_executable(test_wmma_tile tests/test_wmma_tile.cpp)
target_link_libraries(test_wmma_tile PRIVATE amdhip64)

# Phase 0 Spike 2.2: Tiled block GEMM benchmark
add_executable(bench_wmma_gemm tests/bench_wmma_gemm.cpp)
target_link_libraries(bench_wmma_gemm PRIVATE amdhip64)

# Phase 0 Spike 3.1: 4KB sector-aligned direct I/O test
add_executable(test_direct_io tests/test_direct_io.cpp)
target_link_libraries(test_direct_io PRIVATE amdhip64)

# Phase 0 Spike 3.2: Concurrent compute + SDMA transfer overlap test
add_executable(bench_async_overlap tests/bench_async_overlap.cpp)
target_link_libraries(bench_async_overlap PRIVATE amdhip64)

# Phase 0 Spike 4.2: Single-layer toy MoE pipeline with dynamic expert caching
add_executable(test_toy_moe_layer tests/test_toy_moe_layer.cpp)
target_link_libraries(test_toy_moe_layer PRIVATE amdhip64)

# Phase 1 Spike 1.1: DeepSeek-V4 config parser test
add_executable(test_config_parser tests/test_config_parser.cpp)
target_link_libraries(test_config_parser PRIVATE amdhip64)

# Phase 1 Spike 1.2: Safetensors header parser test
add_executable(test_safetensors_parser tests/test_safetensors_parser.cpp)
target_link_libraries(test_safetensors_parser PRIVATE amdhip64)

# Phase 1 Spike 2.1: Wave32 RMSNorm and Clamped SwiGLU test
add_executable(test_swiglu_clamp tests/test_swiglu_clamp.cpp)
target_link_libraries(test_swiglu_clamp PRIVATE amdhip64)

# Phase 1 Spike 2.2: Hyper-Connections Sinkhorn & Post expansion test
add_executable(test_hc_sinkhorn tests/test_hc_sinkhorn.cpp)
target_link_libraries(test_hc_sinkhorn PRIVATE amdhip64)

# Phase 1 Spike 3: Fused INT4 -> FP16 Wave32 WMMA test
add_executable(test_w4a16_wmma tests/test_w4a16_wmma.cpp)
target_link_libraries(test_w4a16_wmma PRIVATE amdhip64)

# Phase 1 Spike 4: Dual-Mode MoE Router test
add_executable(test_moe_router tests/test_moe_router.cpp)
target_link_libraries(test_moe_router PRIVATE amdhip64)

# Phase 1 Spike 4.3: End-to-End DeepSeek-V4 MoE Layer Forward Pass test
add_executable(test_v4_moe_layer tests/test_v4_moe_layer.cpp)
target_link_libraries(test_v4_moe_layer PRIVATE amdhip64)

# Phase 1 Spike 5.1: Sliding-Window Attention & Rotary Embeddings test
add_executable(test_v4_attention tests/test_v4_attention.cpp)
target_link_libraries(test_v4_attention PRIVATE amdhip64)

# Phase 1 Spike 5.2: Complete DeepSeek-V4 Transformer Block Forward Pass test
add_executable(test_v4_block tests/test_v4_block.cpp)
target_link_libraries(test_v4_block PRIVATE amdhip64)

# Phase 1 Spike 6.1: Multi-Layer Pipeline & KV Cache test
add_executable(test_v4_pipeline tests/test_v4_pipeline.cpp)
target_link_libraries(test_v4_pipeline PRIVATE amdhip64)

# Phase 1 Spike 6.2: End-to-End Autoregressive Generation Benchmark
add_executable(bench_v4_generation tests/bench_v4_generation.cpp)
target_link_libraries(bench_v4_generation PRIVATE amdhip64)







