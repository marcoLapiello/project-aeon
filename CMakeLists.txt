cmake_minimum_required(VERSION 3.28)
project(aeon VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

# Build logic lives in cmake/ so this file stays a readable index of what the
# project builds. Target definitions live in `AeonInfrastructure.cmake` — the
# artifact format, storage/streaming tiers, W4A16 expert kernels, the text front
# end and the DeepSeek-V4 graph rewrite.
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(AeonOptions)
include(AeonToolchain)
include(AeonHelpers)

include(AeonInfrastructure)
