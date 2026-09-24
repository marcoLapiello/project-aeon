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

    # Step 6 outcome 5: the swept prefill against its baseline. A speed claim needs
    # its baseline in the same process, and neither arm scales the same way (the
    # sweep's bytes are constant in the prompt, serial's grow linearly), so the bench
    # measures both at several prompt lengths and prints tok/s and bytes for each.
    # The prompt is real natural language, so the tokenizer and encoder are linked in.
    aeon_add_benchmark(bench_prefill_ab
        SOURCES
            tests/bench_prefill_ab.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp)

    # Supply-chain hot-path analysis, Step 1: where a batched window's exposed load
    # goes. Loads the model once, runs the gate-selected strategy per length in a
    # fresh session, and prints the drive / H2D-enqueue / H2D-drain split for prefill
    # and for a short greedy decode. Attribution, not a strategy A/B — see the header.
    aeon_add_benchmark(bench_supply_split
        SOURCES
            tests/bench_supply_split.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp)
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

    # Step 0 oracle: render the artifact's golden vectors and compare byte-for-byte
    # against the checkpoint's own encoder output (checkpoint plan Stage B).
    aeon_add_test(test_dsv4_prompt_encoding_oracle
        SOURCES
            tests/test_dsv4_prompt_encoding_oracle.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
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

    # Routing reuse-distance profiler (Phase 1 of the routing study).
    aeon_add_test(test_routing_reuse SOURCES tests/test_routing_reuse.cpp)
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

# --- Step 6: the layer-major prefill window -----------------------------------
# The plan's Step 6 D-a decides prefill is layer-major within a bounded window.
# This gate certifies the equality half of that decision: a layer-major pass over
# N tokens through the real host (43 layers, the real expert supply) is
# byte-identical to the certified serial path, and the body chunk size is not
# observable. Speed is a separate outcome and is not asserted here.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_prefill_window
        SOURCES tests/test_v4_prefill_window.cpp TIMEOUT 1800)
endif()

# --- Step 6: Warm frozen during prefill (D-b, outcome 3) -----------------------
# Policy A: the prefill sweep must not drain Warm. A gate over the real host and a
# real Warm tier asserts the resident set is identical before and after a prefill,
# that the copy was taken from Warm (not bypassed to NVMe), that leaving the phase
# releases the VRAM copies, and — as the control — that an unfrozen prefill does
# change Warm, so the assertion measures the policy and not the workload.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_warm_frozen_prefill
        SOURCES tests/test_v4_warm_frozen_prefill.cpp TIMEOUT 1800)
endif()

# --- Step 6 item 6: the prefill sweep (bounded drain, layer order, restore) ----
# Prefill and decode are two allocation strategies, so the switch between them is
# asserted as a switch: the sweep frees only what the pass needs, streams whole
# layer sets in layer order, releases each layer as it retires, and restores the
# switch-point set on exit with Warm untouched. The result must stay byte-identical
# to the serial path.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_prefill_sweep
        SOURCES tests/test_v4_prefill_sweep.cpp TIMEOUT 1800)
endif()

# --- Step 4: the routed bank (cached prefill below the sweep's gate) -----------
# A window shorter than the gate runs the same layer-major loop with the route-aware
# cached supply instead of a whole-layer sweep: one layer's worth of the pool is
# drained, the rest preserved, the per-chunk union held to the layer boundary, and
# the switch-point set restored on exit. Byte-identical to serial, and cheaper than a
# sweep below the gate.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_routed_prefill
        SOURCES tests/test_v4_routed_prefill.cpp TIMEOUT 1800)
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

# --- Tier-4 integration: the prefix-cache state contract ----------------------
# Item 22, first half. The plan's gate is "restore is byte-exact with respect to
# never having evicted". `V4Layer::snapshot_state()` existed but nothing restored a
# snapshot and no gate covered it, so the state contract had no executable
# meaning. The instrument is the plan's own: a run the layer never stopped (tokens
# 0..N+K-1) against one that ran the prefix, snapshotted, reset, restored and
# continued, requiring the continuation's tokens and the final state to be
# bit-identical. Boundaries are chosen to be genuinely mid-ratio-window as well as
# on a boundary, so the compressor's in-progress partial state and the wrapped
# local ring both have to survive the round trip. Section C makes the comparison
# load-bearing: a cleared snapshot, a zeroed local ring and lost partial positions
# each have to change the continuation, or B would only show that two runs of the
# same code agree. Not covered, and named in the file: R4 (declining a boundary
# outside the window), tier placement (R5), and the non-token cache-key inputs.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_state_restore
        SOURCES tests/test_v4_state_restore.cpp TIMEOUT 600)
endif()

# --- Real-scale state: window 128 AND index_topk 512, together ----------------
# The plan records that no gate has run the model's own window and its own
# index_topk at the same time: items 16-19 shrink both, and item 20 runs the real
# window with index_topk shrunk to 8. One CSA layer is run for 2200 tokens -
# deliberately past position 2048, because below that a CSA layer commits fewer
# than 512 candidates and select_indexer_topk takes its degenerate "take all"
# path. Section A asserts the selection is *strict* (550 candidates, 512 slots,
# all filled, distinct, in range), so "the top-k was exercised" is a measurement
# rather than an assumption. The assertion is the restore contract at real scale:
# the state is snapshotted mid-run (during the reference run, so it is provably
# the reference's own), restored after a reset, and the continuation must be
# bit-identical to the uninterrupted run. Determinism is structural - no fp64
# oracle, fixed-order accumulation, raw fp16 bit comparison. HCA is deliberately
# not the layer here: it has the real window but no indexer at all (trap 33), so
# it cannot exercise index_topk.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_real_scale_state
        SOURCES tests/test_v4_real_scale_state.cpp TIMEOUT 900)
endif()

# --- Step 3: the HC head reduction -------------------------------------------
# The one graph op that had no code, no oracle and no gate. `hc_head` collapses the
# four residual streams to the single vector every downstream step reads, so an
# error there produces plausibly-scaled logits and fluent-looking garbage. The
# oracle grew `hc_head_reduce` for this gate; the norm's weightlessness, the
# flattened RMS, the scalar scale and the post-sigmoid eps are each asserted as a
# discriminating check rather than by construction.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_hc_head_oracle
        SOURCES tests/test_v4_hc_head_oracle.cpp TIMEOUT 300)
endif()

# --- Routed-expert accumulation ----------------------------------------------
# Replaces the tension between the two old paths: the atomic one cannot fix an
# order (trap 38) and the fp16 read-modify-write one violates plan §2.10.3
# ("accumulate in fp32"). The new pair writes one contribution per expert into its
# own fp32 slice — one writer per element, so determinism is structural — then sums
# in slot order and rounds once. The gate measures the fp16 path's error against
# the same fp64 sum, so it asserts a property rather than a preference.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_moe_accum_oracle
        SOURCES tests/test_v4_moe_accum_oracle.cpp TIMEOUT 300)
endif()

# --- The graph's routed-expert executor (item 23, first step) -----------------
# `V4RoutedExpertExecutor` is the layer body's seam onto the storage system. Until
# this step it had no runtime implementation at all: the only implementations were
# test fixtures and the pre-rewrite `V4Pipeline`'s inline blocks, so the production
# path — supply, promotion, prefetch, leases, staging, then the fused kernels — had
# never run.
#
# The gate holds the kernels fixed and removes storage from one side, so it
# certifies the one thing the executor adds: that the supply path hands the kernels
# the artifact's own bytes for the expert the router selected. The arithmetic is
# Tier 1 item 14/15, item 16 and the item-19b accumulation pair; re-deriving an
# oracle for it here would be a second reference for an already-certified quantity.
# Bit-identity is the right instrument because the only difference between the two
# runs is where the bytes came from.
#
# The pool is deliberately too small for the leases one token accumulates, so the
# executor's capacity fallback (drain, then release) is exercised rather than
# assumed away, and the bit-identity assertion is what shows it is value-neutral.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_expert_executor
        SOURCES tests/test_v4_expert_executor.cpp TIMEOUT 900)
endif()

# --- The graph's head end (P1) ------------------------------------------------
# The composition plan's first build phase: the host that owns the parts, and the
# three ops at the tail of the forward pass — `hc_head` -> final RMSNorm -> LM head,
# fed by the device-side embedding expansion.
#
# Scope, stated because a green line here is easy to over-read. It does NOT
# re-certify `hc_head_wave32_kernel` (tests/test_v4_hc_head_oracle.cpp owns that,
# with 6 of 6 mutations killed); it drives the *device* expansion that gate builds
# on the host, and it composes the two ops nothing has ever run — the final norm
# and the LM head. It is not a forward pass: P1 has no layers, so the residual the
# head consumes is an embedding rather than a 43-layer trajectory.
#
# The instrument for the logits is elementwise against the fp16-rounded oracle, not
# a peak-relative bound alone: at 129280 elements a peak-relative bound hides every
# small logit. A wrong norm or a transposed head moves most of the vector, so the
# count of elements differing from the true value's fp16 rounding is a strong
# check with a measurable floor (~0.2%, from the fp32-vs-fp64 accumulation gap
# straddling a rounding boundary) rather than an arbitrary tolerance.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_graph_head
        SOURCES tests/test_v4_graph_head.cpp TIMEOUT 900)
endif()

# --- The 43-layer driver (P2) -------------------------------------------------
# The composition plan's second build phase: `forward_token` = embedding ->
# 43 x `run_layer` -> head, on the artifact's real dense weights, real embedding
# and real head, with real routed experts delivered through the production
# `V4TieredExpertExecutor` (Hot/Warm/Cold, leases, staging, O_DIRECT). This is the
# first phase whose output is a **token** rather than an intermediate.
#
# Two independent statements, because they are different defects. (1) The
# composition is arithmetically right: every layer's `res_out`, and the head's
# three checkpoints, are compared against `reference::model_body` — the fp64
# oracle this phase adds, written before the driver and pinned by closed-form
# self-checks. (2) The driver is the loop it claims: `forward_token` must produce
# the same 129280 fp16 logits, bit for bit, as the gate's own 43 calls to
# `run_layer` plus the head.
#
# The reference is re-seeded from the *device's* per-step residual and its MoE
# combine is driven by the *device's* selection — the serial-decode gate's method,
# which is the only way a 43-layer comparison stays tight instead of measuring
# accumulated fp16 drift. The selection is checked separately against the device's
# own router logits, per trap 37.
#
# Not covered, and named in the file: the local ring wrap (needs >128 tokens), HCA
# compression (first entry at position 127), the sampler, the text binding, the
# observer and tiering under load. OPENMP is on because the reference is fp64 over
# 43 layers; without it the oracle, not the device, would be the whole cost.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_graph_body
        SOURCES tests/test_v4_graph_body.cpp TIMEOUT 1800 OPENMP)
endif()

# --- The sampler and its seam (P3) --------------------------------------------
# The composition plan's third build phase. The graph ends at logits; this decides
# a token, and it owns the **logit-processor seam** the plan calls non-deferrable
# (structured output and tool-call JSON are logit masks, §6.4) — which is why the
# gate asserts the seam *exists*, not merely that a `sampling` function does.
#
# The gate's clauses, each measured rather than asserted: seeded replay is
# bit-identical; a mask that sets one logit to `-inf` removes that token from the
# support (probability exactly zero, never drawn); `T->0` converges to the argmax
# path; the untruncated defaults reproduce the argmax token. The seam's *ordering*
# is pinned too, by a promotion that must survive `top_k = 1`.
#
# Cheap on purpose, and it says why: the pure sections (the generator, the
# transforms, the seam, determinism) run on hand-built vectors against an
# independently written fp64 reference and need no model at all. The device
# section drives the certified argmax pair, and one section closes the seam on the
# artifact's own logits via `V4Graph::forward_token`. OPENMP is deliberately off —
# there is no fp64 oracle here to parallelise.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_sampler
        SOURCES tests/test_v4_sampler.cpp TIMEOUT 900)
endif()

# --- The text binding (P4) ----------------------------------------------------
# The composition plan's fourth build phase and the plan's **acceptance
# criterion**: one command takes a conversation and returns text. `core/v4_engine.hpp`
# binds the four components that already ran — the artifact tokenizer, the canonical
# prompt encoder, `text::generate_token_ids` and `V4Graph` — plus `V4Sampler`, and the
# artifact's own sampling policy read from `generation_config.json`.
#
# The gate is the acceptance run and it is honest about what it can and cannot see.
# Coherence is not machine-checkable, so the gate asserts what *is*: the engine is
# exactly `forward_token` + `select` (an independent replay of the loop), the reply
# is text (non-empty, decodable, valid UTF-8), a second call is bit-identical, a
# longer conversation gives a different reply than the same user turn alone (so the
# history is genuinely in the context), and every stop reason is reachable. It also
# prints the generated text, because the one claim a human has to judge is the one
# the gate is not entitled to assert.
#
# Where model-backed, it builds the host once and reuses it across sections: the
# assembly is ~8 s and there is no reason to pay it six times.
if(AEON_BUILD_TESTS)
    aeon_add_test(test_v4_engine
        SOURCES
            tests/test_v4_engine.cpp
            src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
            src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp
            src/infrastructure/text/text_generation.cpp
        TIMEOUT 1800)
endif()

# --- Artifact inspection -----------------------------------------------------
# Standalone artifact inspector: dumps the model contract and dense tensor
# inventory (Stage 0 evidence). It uses only kept components (`V4ModelContract`,
# `AeonModelLoader`), so it was promoted out of the legacy gate rather than
# deleted with it.
aeon_add_executable(aeon_model_contract SOURCES tools/aeon_model_contract.cpp)

# --- The text-in/text-out CLI -------------------------------------------------
# The acceptance criterion is one command, and `aeon_chat` is it. It is bound to
# `V4Engine` (the rewritten graph) and built by default, so the default build
# always contains a way to run the model.
#
# It links the text sources because the engine drives them: the tokenizer, the
# canonical prompt encoder (Step 0's port, not the superseded chat formatter) and
# the certified generation loop.
aeon_add_executable(aeon_chat
    SOURCES
        tools/aeon_chat.cpp
        src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
        src/architecture/deepseek_v4/text/dsv4_prompt_encoder.cpp
        src/infrastructure/text/text_generation.cpp)

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
