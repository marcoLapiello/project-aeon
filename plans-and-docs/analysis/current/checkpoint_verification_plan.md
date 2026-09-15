# Checkpoint & Artifact Integrity Plan

## DeepSeek-V4-Flash-0731 INT4 W4A16 — validating the input to the graph rewrite

**Target hardware:** AMD Radeon RX 7900 XTX (RDNA 3, `gfx1100`)
**Artifact:** `models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon`, a byte-identical repack of
`yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16` (INT4 symmetric, group 32, W4A16)
**Companion document:** [Inference Pipeline Plan](inference_pipeline_plan.md) — the graph specification

> **Revision note (2nd revision, 2026-09-15).** This document was rescoped. It previously
> tried to cover the graph as well, and its kernel stages drifted out of date as the graph
> plan advanced. It also contained a factual error that would have rejected a *correct*
> checkpoint (§1.3). What remains is the job the graph plan cannot do: **prove the input
> is sound**, so a graph failure is a graph failure.

---

## 0. Purpose and scope split

The two plans answer different questions and must not be merged:

| Question | Document |
| :--- | :--- |
| Is the **computation** correct? | [Inference Pipeline Plan](inference_pipeline_plan.md) |
| Is the **artifact** sound? | this document |

**Guiding principle (unchanged):** *Do not debug the engine and the checkpoint at the
same time.* A wrong nibble order or a wrong scale dtype produces output that looks exactly
like a broken kernel. You cannot tell them apart by looking at the text; only by looking at
numbers. Verify parsing before kernels.

### In scope

| Stage | Covers | Cost |
| :--- | :--- | :--- |
| **A — Structural audit** | The artifact describes the architecture we think it does | minutes, no GPU |
| **B — Prompt-encoder oracle** | Our formatter reproduces the checkpoint's own encoder on golden vectors | minutes, CPU |
| **C — Repack round-trip** | Our `.aeon` format is lossless w.r.t. the safetensors checkpoint | minutes, CPU |
| **D — Streaming integrity** | The three-tier memory system does not corrupt data in transit | minutes, GPU |

### Explicitly out of scope

- **Original-checkpoint fidelity (old Stages B/C).** Comparing our INT4 artifact against the
  FP4 `deepseek-ai/DeepSeek-V4-Flash-0731` source, and re-deriving transcode SNR. **Decision:**
  not required. The goal is to get the graph *running correctly on the artifact we have*. The
  checkpoint is a swappable component (see §7); if a better W4A16 artifact appears, replacing
  it must be a data change, not a code change. The published fidelity figures (§1.1) are accepted
  as given and used only to set an accuracy ceiling.
- **Kernel numerical verification (old Stage E).** Superseded by the graph plan's Tier 1–23
  gates, which are semantically correct, cite the reference hierarchy, and carry an
  anti-circularity rule. Its kernel list is not maintained here any more.

---

## 1. What is already established about this artifact

Recorded so later gates are checked *against* something. These are the card's claims and the
artifact's own evidence; they are spot-checked in Stage A, not re-derived.

### 1.1 Quantization and scope

- `quantization_config` = `compressed-tensors` / `pack-quantized`, INT4 **symmetric**,
  **group_size 32**, group strategy, minmax observer, weight-only (W4A16) `[V config.json]`.
- Applied **only to routed experts**, matched by
  `re:.*experts\.\d+\.(w1|w2|w3|gate_proj|up_proj|down_proj|gate_up_proj)$` `[V config.json config_groups]`.
- `ignore` list keeps `lm_head`, `embed_tokens`, `norm.*`, `attn.*`, `ffn.gate`, `shared_experts.*`,
  `main_proj`, `(markov_head|confidence_head)` at full precision `[V config.json]`.
- The card states attention, shared experts, router gates, norms, embeddings, `lm_head` and MTP
  heads are untouched, and that FP8→fp16 conversions (390 tensors) are **bit-exact**
  `[V checkpoint README]`.

### 1.2 Fidelity ceiling

The card reports, over all 35,328 transcoded expert tensors, SNR min **22.71 dB**, 1st percentile
23.15 dB, median **25.69 dB**, no range violations `[V checkpoint README]`.

**Consequence for the rewrite:** the artifact is an *approximation* of the source. This bounds
what any parity gate can demand — arguing for tolerances (relative error ~1.5% mean) rather than
bit-exactness against a source we are not using, and it is why the graph plan's gates are stated
as comparisons to an independent reference at a tolerance, not as exact matches.

### 1.3 Correct layer-class expectations `[corrected — the previous revision was wrong here]`

The previous revision said *"layers 2..42 carry compressor + indexer weights"* and instructed:
if indexer tensors appear on layers 0–1, *"the checkpoint was quantized against the wrong
config. Stop."* **That is wrong, and following it would reject a correct checkpoint.**

The correct expectation, established in the graph plan (Tier 0.2f):

| Layer class | `compress_ratio` | Compressor | Indexer |
| :--- | :--- | :--- | :--- |
| Layers 0, 1 | 0 | no | no |
| ratio-4 layers (CSA, even indices 2..42) | 4 | yes | **yes** |
| ratio-128 layers (HCA, odd indices 3..41) | 128 | yes | **no** |

HCA layers compress but do **not** index — they attend all committed compressed rows directly.
Running an indexer on a ratio-128 layer would read `attn.indexer.*` tensors that **do not exist**
in the artifact. This is now a positive assertion in Stage A rather than a stop condition on a wrong
premise.

### 1.4 A working reference run of this exact artifact exists

The card documents an end-to-end run of this W4A16 artifact on 4 `gfx1030` dies with `--dtype
float16`: coherent greedy generation through the chat template, perplexity 2.42 (prose) / 1.72
(code), and a 3,011-token prompt exercising the sparse attention indexer summarized correctly
`[V checkpoint README]`.

**Why this matters:** the graph plan's largest open gate is "compare against a trusted compatible
reference". This is one, for the *same weights*. It runs on a vLLM build carrying the RDNA2
DeepSeek-V4 patches — not upstream vLLM, which has no `rdna`/`gfx11` path. Recording it corrects the
graph plan's caveat from *"no RDNA reference exists"* to *"no upstream RDNA path; a patched build
does exist"* `[V checkpoint README; V vllm/platforms/rocm.py:226]`.

The card also states `--kv-cache-dtype` resolves to **`fp8_ds_mla`**, "the only layout these backends
implement" `[V checkpoint README]` — evidence for the graph plan's open KV fp8 vs bf16 gate.

The card is explicit about its own limits: fp16 only (bf16 untested), memory saving confined to
routed experts, and quality checked for coherence/perplexity rather than by benchmark suite
`[V checkpoint README]`.

---

## 2. Stage A — Structural audit (no GPU, no math)

**Purpose:** confirm the artifact describes the architecture we think it does, before any numerics.

### A.1 Topology
- `num_hidden_layers == 43` `[expect config.json]`.
- `compress_ratios` is `0,0,4,128,4,128,…,4,128,4,0,0,0` — length 43, first two and last three zero.
- **Per layer, assert the §1.3 table:** ratio 0 → no compressor, no indexer; ratio 4 → compressor
  **and** indexer; ratio 128 → compressor **and no indexer**.
- Any other combination stops the audit — but note the corrected consequence: a *missing* indexer on
  a ratio-128 layer is **correct**, not a config mismatch.

### A.2 Tensor inventory
- Parse `model_experts_swizzled.index` and the dense manifest; build name → dtype → shape → location.
- Confirm 43 layers × 256 routed experts, and that the **expert payloads are the only quantized
  tensors** (§1.1 ignore-list check).
- Confirm the dense side carries the full-precision modules: attention (`wq_a`, `q_norm`, `wkv`,
  `kv_norm`, `wq_b`, `wo_a`, `wo_b`), compressor (`fused_wkv_wgate`, `norm`, **`ape`**), indexer
  (`wq_b`, `weights_proj`, its own compressor `ape`/`norm`), `ffn.gate.weight`, `ffn.gate.bias`,
  `ffn.gate.tid2eid` (layers 0–2 only), shared experts, norms, `embed.weight`, `head.weight`,
  `hc_*`, `hc_head_*`, and the `mtp.*` draft stack.
- **Specifically assert two tensors the rewrite depends on and the old plan never listed:**
  - `layers.N.attn.compressor.ape` and `layers.N.attn.indexer.compressor.ape`, fp32, shape
    `[ratio, coff·head_dim]` — required by graph plan §2.4.2.
  - `ffn.gate.bias` present on **layers 3..42 only**, `F32 [256]` — the correction bias; absent on
    the three hash layers, which use `ffn.gate.tid2eid` `I64 [129280, 6]` instead.
- `tie_word_embeddings == false`; `embed.weight` and `head.weight` are distinct `[129280, 4096]`.

### A.3 Deliverable
A checked table: pattern → dtype → shape → layer-class expectation. This is ground truth for Stage C.

---

## 3. Stage B — Prompt-encoder oracle (CPU, no GPU)

**Purpose:** Step 0 of the graph plan (prompt encoding) is *not deferrable* — an "almost right"
template silently changes every prefix and defeats prefix caching. The artifact ships the encoder
and golden vectors, so this gate is exact rather than approximate.

### B.1 The oracle
The checkpoint contains its **own** encoder revision and a test suite:

```
models/…-INT4-W4A16/snapshots/<sha>/encoding/
    encoding_dsv4.py          # 760 lines, revision-specific
    test_encoding_dsv4.py     # harness
    tests/test_input_{1..4}.json   # message lists (incl. tools, thinking, multi-turn)
    tests/test_output_{1..4}.txt   # golden rendered prompts
```

`[V artifact: encoding/ tree]`

> **Use this copy, not vLLM's.** The two differ (760 vs 648 lines; this one carries
> `decode_dsml_to_arguments` and `tool_calls_to_openai_format`). vLLM's
> `deepseek_v4_encoding.py` is a *sibling revision* and is not authoritative for this artifact.
> Cite the artifact's copy for template semantics.

### B.2 Method
- Render each `test_input_N.json` to a prompt string with our formatter.
- Compare **byte-for-byte** against `test_output_N.txt`.
- Cover, at minimum, the four cases the vectors exercise: thinking **with tools** (incl. tool-result
  merge), thinking **without tools** (`drop_thinking` strips earlier reasoning), plus vectors 3 and 4.
- Then tokenize the rendered string and compare token ids against the checkpoint tokenizer for the
  same text — catching an encoder that is byte-right but tokenizer-wrong, or vice versa.

### B.3 Why this is a real gate and not a formality
The graph plan (Tier 0.2e) lists the encoder surface our current formatter **does not implement**:
`developer` role, `latest_reminder` messages, `task` classification tokens, DSML tool-call/tool-result
rendering, `response_format`, the `reasoning_effort` prefix, and the **tools→keep-thinking** rule.
The golden vectors exercise several of these. Expect this gate to **fail on first run** — that is the
point; it is the independent oracle Step 0 currently lacks.

### B.4 Deliverable
A pass/fail per vector, plus the exact rendered diff for any failure. A green Stage B means the
graph's first step is trustworthy.

---

## 4. Stage C — Repacked format round-trip (CPU, no GPU)

**Purpose:** prove the `.aeon` repack is lossless with respect to the safetensors checkpoint it was
made from. This is a pure format transformation — there is no numerical operation, so any difference
is a packing bug, not a precision effect.

### C.1 Round-trip
- Take a tensor from the safetensors checkpoint; run it through the repack; read it back through the
  engine loader; dequantize. **Expected: bit-exact.**
- Extend to every expert, not a sample — the index file has no analogue in the source and therefore
  no external reference, so it is only validated by exhaustively seeking through it.

### C.2 Nibble order
- Synthetic tensor with known values, e.g. `[0,1,2,3,4,5,6,7,-8,…, -1]`.
- Pack with the writer, unpack with the reader, verify the round trip.
- Then assert the **byte layout** explicitly: which nibble holds the even-indexed element. A mismatch
  swaps adjacent elements and produces plausible-looking but wrong output.

### C.3 Scale layout
- Assert scale tensor shape, dtype and ordering contract; element `i` belongs to group `i // 32`, and
  scale index is `i // 32`. A transposed or offset scale array is a smooth, subtle corruption easily
  mistaken for a kernel bug.

### C.4 Deliverable
Pass/fail on C.1–C.3 with the failing element index and both values for any mismatch.

> **Note.** `tests/test_aeon_loader.cpp` and `tests/test_aeon_swizzled_loader.cpp` already assert
> format version, layer/expert counts and sector size, and
> `scripts/convert_safetensors_to_aeon.py::verify_swizzled_expert_payload` performs a de-swizzle
> round-trip check at conversion time. Stage C's job is to make the *numerical* round-trip an explicit,
> independently runnable gate rather than an assertion inside the converter.

---

## 5. Stage D — Streaming and pool integrity (GPU)

**Purpose:** prove the three-tier hierarchy is transparent to the numerics.

### D.1 Expert integrity across tiers
- Load an expert to VRAM, dequantize, record.
- Evict, reload from host RAM, dequantize, compare. **Expected: bit-exact.**
- Evict, reload from NVMe via `O_DIRECT`/`io_uring`, dequantize, compare. **Expected: bit-exact.**

### D.2 Concurrent streaming
- Run a forward pass while experts stream in; verify no expert is read while partially written, and
  that index positions stay valid under concurrent access.

### D.3 Deliverable
Confirmation that tier placement does not change a single bit — i.e. any numerical difference is
attributable to the graph, never to the memory system.

---

## 6. Failure triage

| Symptom | Most likely cause | Check |
| :--- | :--- | :--- |
| Prompt differs from golden in tools/thinking | Missing encoder surface (developer/tasks/DSML/reasoning_effort) | Stage B |
| Output fluent but semantically wrong | Wrong nibble order, or scale offset | C.2, C.3 |
| Output correct short, wrong long | Indexer/compressor state error — **or a 4-bit artifact limitation** | graph plan; §1.2 |
| Identical artifact, different result on reload | Streaming/caching corruption | Stage D |
| Output random tokens | Wrong dequant table, or wrong layer count | A.1, C.1 |
| Parity error just above tolerance | 4-bit approximation floor, not a bug | §1.2 |

---

## 7. Checkpoint portability (a design requirement, not a gate)

The stated goal is to get the graph running correctly, and to be able to swap in a better W4A16
artifact with little effort. Two consequences for the rewrite:

1. **The graph must not encode artifact-specific facts.** Op semantics come from the config and the
   specification; anything that is a property of *this* checkpoint (its quantization scheme, its
   expert payload layout, its tensor names) belongs behind the backend/descriptor boundary that
   [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md)
   already establishes.
2. **Do not tune tolerances or expectations to this artifact's fidelity floor.** §1.2's numbers are
   context for interpreting a parity result, not a target to fit. A tolerance that merely accommodates
   this checkpoint will hide the same bug in the next one.

---

## 8. Execution order

1. **Stage A** — structural audit. Minutes, no compute.
2. **Stage B** — prompt-encoder oracle. Minutes, CPU. **Highest value: it gates Step 0 and currently
   has no independent check.**
3. **Stage C** — format round-trip. Minutes, CPU.
4. **Stage D** — streaming integrity. Requires the runtime; run alongside the graph's own gates.

Stages A and B need no GPU and can run before any Tier-1 work. Stage C is independent of the graph.
Stage D belongs with the graph's integration tests.

**This document does not gate the graph rewrite's start** — with one exception: **Step 0 must not be
implemented from memory**, because the artifact ships its own encoder. Stage B is the check.
