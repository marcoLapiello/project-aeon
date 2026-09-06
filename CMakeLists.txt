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
