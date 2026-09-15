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
            src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp
            src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp)

    # Step 0 oracle: render the artifact's golden vectors and compare byte-for-byte
    # against the checkpoint's own encoder output (checkpoint plan Stage B).
    aeon_add_test(test_dsv4_prompt_encoding_oracle
        SOURCES
            tests/test_dsv4_prompt_encoding_oracle.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp
            src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp)

    # EOS-aware token generation loop and stop reasons (CPU-side).
    aeon_add_test(test_text_generation
        SOURCES
            tests/test_text_generation.cpp
            src/infrastructure/text/text_generation.cpp)
endif()

# --- Expert supply and telemetry ---------------------------------------------
if(AEON_BUILD_TESTS)
    # Dynamic memory budget and global hot pool.
    aeon_add_test(test_dynamic_expert_pool SOURCES tests/test_dynamic_expert_pool.cpp)

    # Persistent Warm ownership, pending transfers, and leases.
    aeon_add_test(test_expert_registry_warm_state SOURCES tests/test_expert_registry_warm_state.cpp)

    # Versioned, phase-labelled supply telemetry JSONL.
    aeon_add_test(test_supply_telemetry SOURCES tests/test_supply_telemetry.cpp)

    # Routing profile parser and checkpoint validation.
    aeon_add_test(test_routing_profile SOURCES tests/test_routing_profile.cpp)
endif()

# --- Tier-1 primitives: independent-oracle gates -----------------------------
# The rewrite certifies graph primitives one at a time, each against a host fp64
# reference written from the specification and sharing no code with the kernel
# (plan Part V, anti-circularity rule). These targets belong to the rewrite, not
# the legacy graph, so they stay in the default build.
if(AEON_BUILD_TESTS)
    # Step 2.1 — RMSNorm, weighted and unit forms, versus reference/dsv4_oracle.hpp.
    aeon_add_test(test_v4_norm_oracle SOURCES tests/test_v4_norm_oracle.cpp)

    # Step 2.3 — RoPE forward and inverse, two bases, tail-only rotation, versus
    # the same oracle. Asserts the discriminating properties, not just closeness.
    aeon_add_test(test_v4_rope_oracle SOURCES tests/test_v4_rope_oracle.cpp)
endif()

# --- Kept model-side components ----------------------------------------------
# These validate components the rewrite keeps, against references written
# independently of the code under test. They are promoted out of the legacy gate
# because they are not coupled to the pre-rewrite layer schedule.
if(AEON_BUILD_TESTS)
    # Checkpoint contract: 43-layer schedule, tensor names and shapes, hash/biased
    # router split. Validates the artifact, not the graph.
    aeon_add_test(test_v4_model_contract SOURCES tests/test_v4_model_contract.cpp)

    # Dual-mode MoE router: sqrt(softplus) scoring, bias on scores, hash layers
    # with no bias, flat top-6. Matches the reference (Tier 0.2d).
    aeon_add_test(test_moe_router SOURCES tests/test_moe_router.cpp)

    # Hyper-Connections Sinkhorn and stream expansion, against an inline host
    # reference. Covers the 3-entry HC scale.
    aeon_add_test(test_hc_sinkhorn SOURCES tests/test_hc_sinkhorn.cpp)
endif()
