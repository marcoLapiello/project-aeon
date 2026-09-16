# Project Aeon — build options.
#
# The rewrite is developed on its own branch and owns the default build. The
# pre-rewrite DeepSeek-V4 graph (core/, kernels/, reference/), its graph-specific
# tests, and the tools/CLI that drive it are gated behind an option so they can
# never be compiled or run by accident. They are kept reachable only for
# reference cross-checks while the rewrite is in progress.

option(AEON_ENABLE_LEGACY_V4_GRAPH
    "Build the pre-rewrite DeepSeek-V4 graph, its graph-specific tests, and the legacy CLI/tools."
    OFF)

option(AEON_BUILD_TESTS
    "Register the infrastructure and backend CTest targets."
    ON)

option(AEON_BUILD_BENCHMARKS
    "Build the infrastructure and backend benchmark targets."
    ON)

# Worker threads for the oracle gates that opt into host OpenMP
# (`aeon_add_test(... OPENMP)`). Those gates alternate between fp64 oracle work,
# which spreads cleanly over cores, and device work that does not.
#
# Defaults to the **physical** core count rather than letting libomp pick, which
# is the logical count and therefore doubles up on SMT siblings. Measured on the
# 32-core / 64-thread dev machine, on the item-17 gate: 16 threads 61.1 s,
# 32 threads 52.3 s, 64 threads (the libomp default) 63.5-68.8 s. The SMT
# siblings cost ~20% because the worker threads contend with the HIP runtime's
# own host-side threads. Set to 0 to leave `OMP_NUM_THREADS` unset and go back to
# the libomp default.
if(NOT DEFINED AEON_OPENMP_THREADS)
    cmake_host_system_information(RESULT _aeon_physical_cores QUERY NUMBER_OF_PHYSICAL_CORES)
    if(NOT _aeon_physical_cores)
        set(_aeon_physical_cores 0)
    endif()
    set(AEON_OPENMP_THREADS ${_aeon_physical_cores} CACHE STRING
        "Worker threads for the OpenMP-parallel oracle gates (0 = libomp default).")
    unset(_aeon_physical_cores)
endif()
