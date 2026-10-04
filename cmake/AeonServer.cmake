# Project Aeon — the G5 serving layer.
#
# One native binary serves a conversation over HTTP, streamed out, queued when
# busy and cancellable. The transport is a vendored header-only HTTP library with
# no OpenSSL or zlib, so it adds no second dependency stack (R7); it is built with
# `-w` because third-party warnings are not our build's signal.
#
# `aeon_httplib`, `aeon_server_core` and every `server/` target deliberately link
# **no** HIP runtime: anything in the serving layer that could name the engine
# would pull HIP in, and the host tests below failing to link `amdhip64` is the
# build-level proof that G5 depends on the engine only through the neutral
# conversation seam. `aeon_server_core` does link the one G1 source it needs
# (`text_generation.cpp`, for `stop_reason_name`), which is pure CPU and drags in
# no device runtime.

if(AEON_BUILD_SERVER)

find_package(Threads REQUIRED)

add_library(aeon_httplib STATIC third_party/cpp-httplib/httplib.cpp)
target_compile_options(aeon_httplib PRIVATE -w)
# Without TCP_NODELAY, SSE chunks wait on Nagle's algorithm and streaming stops
# looking streamed.
target_compile_definitions(aeon_httplib PRIVATE CPPHTTPLIB_TCP_NODELAY=1)
target_link_libraries(aeon_httplib PUBLIC Threads::Threads)
target_include_directories(aeon_httplib PUBLIC third_party/cpp-httplib)

# The serving layer itself: the queue/worker service, the codec and the HTTP
# transport, plus the pure G1 generation source they transitively name.
add_library(aeon_server_core STATIC
    server/conversation_service.cpp
    server/openai_codec.cpp
    server/http_server.cpp
    src/infrastructure/text/text_generation.cpp)
target_link_libraries(aeon_server_core PUBLIC aeon_httplib)
target_include_directories(aeon_server_core PUBLIC ${CMAKE_SOURCE_DIR})

# aeon_add_host_test(<name> SOURCES <files...> [TIMEOUT <seconds>])
# A CTest that links the server library but **not** `amdhip64`. Running from
# `${CMAKE_SOURCE_DIR}` matches `aeon_add_test`, so a test may still open
# artifacts by relative path.
function(aeon_add_host_test name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "SOURCES" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "aeon_add_host_test(${name}): SOURCES is required")
    endif()
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE aeon_server_core)
    add_test(NAME ${name} COMMAND ${name})
    set(properties WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    if(ARG_TIMEOUT)
        list(APPEND properties TIMEOUT ${ARG_TIMEOUT})
    endif()
    set_tests_properties(${name} PROPERTIES ${properties})
endfunction()

# --- the serving binary (the composition root) --------------------------------
# This is the one target that names both the engine and the serving layer; it links
# `amdhip64` through `aeon_add_executable`, which is exactly why `server/` must not.
aeon_add_executable(aeon_serve
    SOURCES
        tools/aeon_serve.cpp
        src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
        src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp)
target_link_libraries(aeon_serve PRIVATE aeon_server_core)
target_include_directories(aeon_serve PRIVATE ${CMAKE_SOURCE_DIR})

# R7: the product installs as one piece.
install(TARGETS aeon_serve aeon_chat aeon_info RUNTIME DESTINATION bin)

if(AEON_BUILD_TESTS)
    # Queue, worker and event streams, driven by a fake engine (no GPU, no model).
    aeon_add_host_test(test_server_service SOURCES tests/test_server_service.cpp)

    # OpenAI request parsing and response writing (no GPU, no model).
    aeon_add_host_test(test_server_codec SOURCES tests/test_server_codec.cpp)

    # The HTTP transport, streaming, queueing and cancellation (no GPU, no model).
    aeon_add_host_test(test_server_http SOURCES tests/test_server_http.cpp TIMEOUT 300)

    # The layering gate: G5 must not name a model, format, kernel or HIP type.
    add_test(NAME check_server_layering
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_server_layering.sh)
    set_tests_properties(check_server_layering PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

endif()
