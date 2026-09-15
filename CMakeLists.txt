cmake_minimum_required(VERSION 3.28)
project(aeon VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

# Build logic lives in cmake/ so this file stays a readable index of what the
# project builds. Target definitions are split by ownership:
#
#   AeonInfrastructure.cmake  — format, storage, backend kernels, text front end
#   AeonLegacyGraph.cmake     — the pre-rewrite graph, gated OFF by default
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(AeonOptions)
include(AeonToolchain)
include(AeonHelpers)

include(AeonInfrastructure)
include(AeonLegacyGraph)

message(STATUS "Project Aeon: legacy V4 graph ${AEON_ENABLE_LEGACY_V4_GRAPH}")
