# Project Aeon — infrastructure, backend, and text targets.
#
# These targets do not depend on the model graph. They validate the artifact
# format, the storage/streaming tiers, the W4A16 expert kernels, and the text
# front end — assets the graph rewrite reuses rather than replaces. They stay in
# the default build on every branch.

# --- Toolchain sanity --------------------------------------------------------
add_executable(smoke_check src/smoke.cpp)
target_link_libraries(smoke_check PRIVATE amdhip64)

# --- Hardware inspection -----------------------------------------------------
# Topology and device discovery.
aeon_add_executable(aeon_info SOURCES tools/aeon_info.cpp)

# --- Storage and streaming ---------------------------------------------------
# Model-backed batched O_DIRECT expert reads.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_model_direct_io SOURCES tests/test_model_direct_io.cpp)
endif()

if(AEON_BUILD_BENCHMARKS)
    # Request shape and queue-width investigation.
    aeon_add_benchmark(bench_model_direct_io SOURCES tests/bench_model_direct_io.cpp)
endif()

# --- Backend kernels and artifact format -------------------------------------
if(AEON_BUILD_TESTS)
    # Wave32 RMSNorm and clamped SwiGLU.
    aeon_add_test(test_swiglu_clamp SOURCES tests/test_swiglu_clamp.cpp)

    # Parallel W4A16 swizzle layout contract (host-side).
    aeon_add_test(test_w4a16_swizzle SOURCES tests/test_w4a16_swizzle.cpp)

    # Swizzled W4A16 decode GEMV (silicon).
    aeon_add_test(test_w4a16_swizzled_gemv SOURCES tests/test_w4a16_swizzled_gemv.cpp)

    # Fused swizzled W1/W3 GEMV (silicon).
    aeon_add_test(test_w4a16_swizzled_dual_gemv SOURCES tests/test_w4a16_swizzled_dual_gemv.cpp)

    # Fused W2 weighted accumulation (silicon).
    aeon_add_test(test_aeon_moe_fused_w2 SOURCES tests/test_aeon_moe_fused_w2.cpp)

    # Native `.aeon` loader.
    aeon_add_test(test_aeon_loader SOURCES tests/test_aeon_loader.cpp)

    # Version-2 swizzled `.aeon` loader.
    aeon_add_test(test_aeon_swizzled_loader SOURCES tests/test_aeon_swizzled_loader.cpp)
endif()

# --- Text front end ----------------------------------------------------------
if(AEON_BUILD_TESTS)
    # Native DSV4 tokenizer artifact and ByteLevel-BPE parity (CPU-side).
    aeon_add_test(test_dsv4_tokenizer
        SOURCES
            tests/test_dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp)

    # Native DSV4 chat/thinking prompt formatting (CPU-side).
    aeon_add_test(test_dsv4_chat_formatter
        SOURCES
            tests/test_dsv4_chat_formatter.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp)

    # EOS-aware token generation loop and stop reasons (CPU-side).
    aeon_add_test(test_text_generation
        SOURCES
            tests/test_text_generation.cpp
            src/infrastructure/text/text_generation.cpp)
endif()

# --- Expert supply and telemetry ---------------------------------------------
if(AEON_BUILD_TESTS)
    # Persistent Warm ownership, pending transfers, and leases.
    aeon_add_test(test_expert_registry_warm_state SOURCES tests/test_expert_registry_warm_state.cpp)

    # Versioned, phase-labelled supply telemetry JSONL.
    aeon_add_test(test_supply_telemetry SOURCES tests/test_supply_telemetry.cpp)

    # Routing profile parser and checkpoint validation.
    aeon_add_test(test_routing_profile SOURCES tests/test_routing_profile.cpp)
endif()
