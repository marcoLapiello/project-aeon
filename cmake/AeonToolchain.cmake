# Project Aeon — toolchain, ROCm discovery, compile flags, and global layout.
#
# Included from the top-level CMakeLists.txt after AeonOptions.cmake. This module
# owns everything that applies to every target; per-target definitions live in
# AeonInfrastructure.cmake.

# --- C++20 -------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# --- ROCm installation discovery ---------------------------------------------
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

# Target AMD GPU architecture (Navi 31 / RX 7900 series).
set(AEON_GPU_TARGET "gfx1100" CACHE STRING "Target AMD GPU architecture")
message(STATUS "Project Aeon: Offload target architecture: ${AEON_GPU_TARGET}")

# --- Compiler flags ----------------------------------------------------------
# -mno-wavefrontsize64 enforces Wave32 execution mode on RDNA3.
add_compile_options(
    -O3
    -Wall
    -Wextra
    -Wno-unused-parameter
    --offload-arch=${AEON_GPU_TARGET}
    -mno-wavefrontsize64
)

# --- Global include and link directories -------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${ROCM_PATH}/include
)
link_directories(
    ${ROCM_PATH}/lib
)

# --- Output layout -----------------------------------------------------------
# All binaries land in <build>/bin.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Build identity ----------------------------------------------------------
# Embedded into tools that record provenance (routing profiles, evidence).
execute_process(
    COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short HEAD
    OUTPUT_VARIABLE AEON_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT AEON_GIT_COMMIT)
    set(AEON_GIT_COMMIT "unknown")
endif()
