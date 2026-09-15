# Project Aeon — shared target helpers.
#
# Every Aeon executable links the HIP runtime. Every CTest runs from the
# repository root so model-backed tests can open `.aeon` artifacts by relative
# path, which is what the previous hand-written target block did implicitly.

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

# aeon_add_test(<name> SOURCES <files...> [TIMEOUT <seconds>])
# Builds the target, registers it with CTest, and pins its working directory.
function(aeon_add_test name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "SOURCES" ${ARGN})
    aeon_add_executable(${name} SOURCES ${ARG_SOURCES})
    add_test(NAME ${name} COMMAND ${name})
    set(properties WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    if(ARG_TIMEOUT)
        list(APPEND properties TIMEOUT ${ARG_TIMEOUT})
    endif()
    set_tests_properties(${name} PROPERTIES ${properties})
endfunction()

# aeon_add_benchmark(<name> SOURCES <files...>)
# Benchmarks are built but intentionally not registered as CTest cases.
function(aeon_add_benchmark name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    aeon_add_executable(${name} SOURCES ${ARG_SOURCES})
endfunction()
