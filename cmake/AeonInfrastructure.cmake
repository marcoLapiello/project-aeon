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

    # Step 2.2 — MLA Q/KV paths. A composition gate: replays the pipeline's kernel
    # order and compares every intermediate, so a wrong wiring (trap 5) fails even
    # though each kernel is individually correct.
    aeon_add_test(test_v4_mla_oracle SOURCES tests/test_v4_mla_oracle.cpp)

    # Steps 2.0 / 2.7 — Hyper-Connections. The audit found a structural error
    # here, so the gate asserts the discriminating properties: the comb index
    # convention (a transposed comb is still doubly stochastic), the third
    # hc_scale entry, the 20-iteration count, and the asymmetric eps placement.
    aeon_add_test(test_v4_hc_oracle SOURCES tests/test_v4_hc_oracle.cpp)

    # Step 2.4.1 — attention score + sink + softmax. Asserts the exact local
    # window boundary, the sink as a denominator-only term, and the full-head
    # 1/sqrt(512) scale.
    aeon_add_test(test_v4_attention_sink_oracle
        SOURCES tests/test_v4_attention_sink_oracle.cpp)

    # Step 2.4.2 — compressor + APE, both ratio classes (4 with overlap, 128
    # without). Asserts that APE is a score-only term, the APE row periodicity,
    # the window length, the two-segment overlap mapping, and the RoPE position.
    aeon_add_test(test_v4_compressor_oracle
        SOURCES tests/test_v4_compressor_oracle.cpp)

    # Step 2.4.3 — lightning indexer + top-k, and the Gate 11 measurement that
    # settles the Hadamard rotation question empirically.
    aeon_add_test(test_v4_indexer_oracle SOURCES tests/test_v4_indexer_oracle.cpp)

    # Step 2.5 — grouped output projection (wo_a low-rank + wo_b). Asserts the
    # per-group structure by construction: group isolation under perturbation,
    # and that the group-major weight layout is load-bearing.
    aeon_add_test(test_v4_grouped_wo_oracle
        SOURCES tests/test_v4_grouped_wo_oracle.cpp)

    # Step 2.9 — MoE router. Asserts the post-softplus bias, the flat top-6 (no
    # n_group/topk_group), the lowest-index tie-break, and the hash/biased layer
    # split — the last against the artifact's own tid2eid table, at real token ids.
    aeon_add_test(test_v4_router_oracle SOURCES tests/test_v4_router_oracle.cpp)

    # Steps 2.10.2 / 2.10.3 — routed expert: fused INT4 dequant, matmul, and the
    # asymmetric clamped SwiGLU. Certifies the format (signed zero point, nibble
    # permutation), the clamp rule, and the composed FFN against an fp64 oracle,
    # then re-checks W1 and the whole FFN on a real artifact payload.
    aeon_add_test(test_v4_expert_oracle SOURCES tests/test_v4_expert_oracle.cpp)

    # Step 2.10.4 — shared expert: dense fp16 FFN, no routing, same clamp. The
    # clamp rule itself is certified at item 14; this certifies the structure,
    # the numerics on real artifact tensors, and that the combine applies the
    # shared contribution exactly once.
    aeon_add_test(test_v4_shared_expert_oracle
        SOURCES tests/test_v4_shared_expert_oracle.cpp)
endif()

# --- Tier-2 composition: one full layer, versus the composed oracle ----------
# Tier 1 certified the primitives; a layer fails in the wiring between them, so
# this gate drives the layer body itself (`core/v4_layer_body.hpp`) on the
# artifact's real layer-0 weights and compares `res_out` and every intermediate
# checkpoint against the fp64 composition in `reference/dsv4_oracle.hpp`.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_layer_body_oracle
        SOURCES tests/test_v4_layer_body_oracle.cpp)
endif()

# --- Tier-2 composition: the compressed attention classes ---------------------
# Item 17. The same body as above, on the ratio-4 (CSA) and ratio-128 (HCA)
# layers, which is where trap 33 lives: only CSA selects compressed rows through
# the indexer; HCA attends every committed compressed row and has no indexer.
# OPENMP: the oracle's fp64 GEMV is ~90% of this gate's cost and is spread
# cleanly over cores (see aeon_enable_openmp); 136 s single-threaded.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_layer_body_compressed_oracle
        SOURCES tests/test_v4_layer_body_compressed_oracle.cpp OPENMP)
endif()

# --- Tier-2 composition: the serial loop --------------------------------------
# Item 18. Items 16/17 both overwrite the device's residual with the oracle's
# after every step, so they measure one layer's composition and are blind to what
# a loop does. This drives a three-layer stack (Sliding, CSA, HCA) for 136 tokens
# with the device carrying its own residual — across 34 CSA boundaries and one
# HCA boundary — and compares the whole accumulated state, not just the step.
# OPENMP: 408 oracle layer-bodies, 203 s single-threaded (see aeon_enable_openmp).
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_layer_body_serial_oracle
        SOURCES tests/test_v4_layer_body_serial_oracle.cpp TIMEOUT 600 OPENMP)
endif()

# --- Tier-3 sequence: chunked batched prefill --------------------------------
# Item 19. Drives the same layer body a chunk at a time and requires the result to
# be bit-identical to the same tokens run one at a time, for three attention
# classes and four chunk schedules. The chunk's keys are held in a per-chunk
# buffer and a per-query row-set is composed from the ring plus that buffer,
# because writing the chunk into the ring first evicts keys its own early queries
# need (trap 39) — section B asserts the ring is untouched during a chunk.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_layer_body_chunk_oracle
        SOURCES tests/test_v4_layer_body_chunk_oracle.cpp TIMEOUT 600)
endif()

# --- Tier-3 sequence: the long-context lifecycle ------------------------------
# Item 20. Every earlier layer-body gate shrinks the local window (to 6, 10, 4)
# and the index top-k so a wrap fits in a short run, and each names the shrinkage
# as uncovered. This one runs at the model's own window (128) for 260 tokens —
# long enough to reuse the ring twice — and asserts *closed forms and invariants*
# rather than comparing checkpoints against an fp64 reference: the ring's
# contents are predicted from the token count, the two stores must be
# bit-identical under each other's writes, and the compressed capacity is shown to
# be exactly what the declared context produces. Section E demonstrates why the
# refusal to take a position past capacity is load-bearing rather than decorative
# (trap 40): a wrapped compressed ring is invisible to the kernel's own position
# guard, so it would silently turn HCA's "every committed row" into "the newest
# K". No OPENMP: with no fp64 oracle there is nothing to spread over cores.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_layer_body_lifecycle
        SOURCES tests/test_v4_layer_body_lifecycle.cpp TIMEOUT 600)
endif()

# --- Tier-4 integration: streaming / tiering ---------------------------------
# Item 21. Gate: expert bytes bit-exact across Hot / Warm / Cold. This is the
# first gate to drive `TieredExpertSupply` itself — the code that decides which
# tier answers a request, which staging slot an I/O lands in, which host slot a
# demotion writes to, and which stream a copy is enqueued on. The two existing
# storage tests cover the legs in isolation (a standalone O_DIRECT read, and one
# hand-driven H2D), so neither could see a defect in those decisions. Every leg
# is compared against the mmapped container — an I/O path none of the three
# produced — and each delivery's *tier* is asserted as well as its bytes, because
# a silently dropped demotion would answer from Cold and pass a byte check while
# measuring nothing.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_expert_tiering
        SOURCES tests/test_v4_expert_tiering.cpp TIMEOUT 300)
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
