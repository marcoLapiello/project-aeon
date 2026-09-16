# Project Aeon — shared target helpers.
#
# Every Aeon executable links the HIP runtime. Every CTest runs from the
# repository root so model-backed tests can open `.aeon` artifacts by relative
# path, which is what the previous hand-written target block did implicitly.

# aeon_enable_openmp(<target>)
# Opts a target into host OpenMP parallelism. The reference oracle
# (`reference/dsv4_oracle.hpp`) is fp64 and single-threaded by construction: its
# reductions are dependent-add chains, so one core reaches ~2 GFLOP/s and the
# multi-token gates take minutes. The parallelization lives in the oracle and is
# guarded by `_OPENMP`, so this switch is the *entire* difference between a target
# taking the serial path and the parallel one; a target that does not call this
# compiles the original loop verbatim.
#
# `-fopenmp` is passed to both compile and link (the clang driver pulls in its own
# runtime, so no separate library is named). The AMDGPU offload flags are already
# global, and this toolchain accepts the two together — the pragma only ever
# appears in host code, so the device pass is unaffected.
function(aeon_enable_openmp target)
    target_compile_options(${target} PRIVATE -fopenmp)
    target_link_options(${target} PRIVATE -fopenmp)
endfunction()

# aeon_add_executable(<name> SOURCES <files...> [DEFINES <defs...>])
function(aeon_add_executable name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEFINES" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "aeon_add_executable(${name}): SOURCES is required")
    endif()
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE amdhip64)
    if(ARG_DEFINES)
        target_compile_definitions(${name} PRIVATE ${ARG_DEFINES})
    endif()
endfunction()

# aeon_add_test(<name> SOURCES <files...> [TIMEOUT <seconds>] [OPENMP])
# Builds the target, registers it with CTest, and pins its working directory.
# `OPENMP` opts the target into the oracle's host parallel path (see above).
function(aeon_add_test name)
    cmake_parse_arguments(ARG "OPENMP" "TIMEOUT" "SOURCES" ${ARGN})
    aeon_add_executable(${name} SOURCES ${ARG_SOURCES})
    if(ARG_OPENMP)
        aeon_enable_openmp(${name})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set(properties WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    if(ARG_TIMEOUT)
        list(APPEND properties TIMEOUT ${ARG_TIMEOUT})
    endif()
    if(ARG_OPENMP AND AEON_OPENMP_THREADS GREATER 0)
        list(APPEND properties ENVIRONMENT "OMP_NUM_THREADS=${AEON_OPENMP_THREADS}")
    endif()
    set_tests_properties(${name} PROPERTIES ${properties})
endfunction()

# aeon_add_benchmark(<name> SOURCES <files...>)
# Benchmarks are built but intentionally not registered as CTest cases.
function(aeon_add_benchmark name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    aeon_add_executable(${name} SOURCES ${ARG_SOURCES})
endfunction()
