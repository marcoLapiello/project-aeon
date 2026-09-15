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
