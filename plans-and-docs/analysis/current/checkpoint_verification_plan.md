# Checkpoint Verification Plan
## DeepSeek-V4-Flash-0731 INT4 W4A16 — Validating the Source Checkpoint and the Repacked Format

**Target hardware:** AMD Radeon RX 7900 XTX (RDNA 3, gfx1100)
**Source checkpoint:** `deepseek-ai/DeepSeek-V4-Flash-0731` (FP4 e2m1 experts, FP8 e4m3 dense)
**Working checkpoint:** `yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16` (INT4 symmetric, group 32, W4A16)
**Goal:** Prove, in isolation, that (a) the source checkpoint is internally consistent, (b) the community INT4 checkpoint is a faithful transcode of it, and (c) your repacked byte-identical format preserves both.

---

## 0. Guiding Principle

Do not debug the engine and the checkpoint at the same time. Every stage below is designed to be **independently falsifiable**: each produces a numeric artifact that either matches a documented reference or does not. If a stage fails, you stop and fix that stage. You never proceed to the next stage on a failed one.

The single most important rule: **verify format parsing before verifying kernels.** A wrong nibble order or a wrong scale dtype produces output that looks exactly like a broken kernel. You cannot tell them apart by looking at the text output. You can only tell them apart by looking at numbers.

---

## 1. Stage A — Static Structural Audit (no GPU, no math)

**Purpose:** Confirm the checkpoint describes the architecture you think it does.

### A.1 Layer count and topology
- Read `config.json`. Confirm `num_hidden_layers == 43`.
- Confirm the layer-type array: layers 0 and 1 must be sliding-window-attention-only, with **no** compressor and **no** indexer weights present in the index file.
- Confirm layers 2..42 carry compressor + indexer weights.
- If the config says 61 layers, or if the index file contains indexer tensors for layers 0-1, the checkpoint was quantized against the wrong config. **Stop.**

### A.2 Tensor inventory
- Parse the safetensors index (`model.safetensors.index.json`).
- Build a table of every tensor: name, dtype, shape, shard.
- Group by module: embeddings, norms, attention (q/k/v/o), compressor, indexer, router gate, shared experts, routed experts (w1/w2/w3 or gate/up/down), `lm_head`, MTP heads.
- Confirm the routed expert count is 256 per layer and that the expert tensors are the only ones in the quantized dtype.
- Confirm attention, shared experts, router gates, norms, embeddings, `lm_head`, and MTP heads are **not** in the INT4 dtype (they should be fp16/bf16/fp8 depending on the checkpoint).

### A.3 Quantization metadata
- Locate the quantization config (`quantization_config` in `config.json`, or a `quantize_config.json`).
- Record exactly: `quant_method`, `format` (`pack-quantized`), `num_bits` (4), `group_size` (32), `symmetric` (true), `actorder`, `damp_percent`, `desc_act`.
- Record the scale dtype and the zero-point presence.
- **Write these down verbatim.** Every later stage is checked against this record.

### A.4 Deliverable
A single table: tensor name pattern → dtype → shape → expected quantization parameters. This is your ground truth for the rest of the plan.

---

## 2. Stage B — Source Format Ground Truth (CPU only)

**Purpose:** Establish what the *original* FP4 checkpoint actually contains, independent of any community conversion.

### B.1 The FP4 lookup table
The source experts are FP4 e2m1 packed two-per-byte as `e2m1fn_x2`, with a per-block UE8M0 scale. The code-to-value mapping is:

```
FP4_TABLE = [ 0,  0.5,  1,  1.5,  2,  3,  4,  6,
              0, -0.5, -1, -1.5, -2, -3, -4, -6 ]
```

Structure to verify explicitly:
- The **high bit is the sign bit**.
- The low three bits index the magnitude.
- The first eight entries are positive, the second eight are negative.
- The magnitudes are **non-uniform**: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

### B.2 UE8M0 scale decoding
- UE8M0 is an unsigned 8-bit exponent-only format: value = `2^(e - 127)`.
- There is no mantissa and no sign. Confirm your decoder produces powers of two only.
- Confirm the block size the scale applies to (this is the source block size, which may differ from the INT4 group size of 32).

### B.3 Dequantize one expert from the source
- Pick a single expert tensor, e.g. layer 20, expert 137, `w1`.
- Dequantize it fully on CPU using B.1 and B.2.
- Record: min, max, mean, std, and a 64-bin histogram.
- Record the count of exact zeros (FP4 has two zero codes, `0b0000` and `0b1000` — both map to 0.0).

### B.4 Deliverable
A reference tensor and its statistics. This is the **only** ground truth in the entire plan. Everything else is compared to it.

---

## 3. Stage C — Community Checkpoint Fidelity (CPU only)

**Purpose:** Determine whether the INT4 checkpoint is a faithful transcode of the source.

### C.1 Dequantize the same expert from the INT4 checkpoint
- Apply the documented scheme: INT4 symmetric, group size 32, `pack-quantized`.
- Dequantization is `w = q * scale`, where `q` is the signed 4-bit integer in `[-8, 7]` and `scale` is per-group.
- Record the same statistics as B.3.

### C.2 Compare against the source
- Compute per-element error `e = w_int4 - w_fp4`.
- Compute SNR: `20 * log10(||w_fp4|| / ||e||)`.
- **Expected:** SNR between 22 and 27 dB. The `yiminyuan` card reports min 22.71 dB, median 25.69 dB across all 35,328 expert tensors.
- Compute relative L2 error. **Expected:** roughly 1.5% mean.
- If SNR is below ~20 dB, or if the error distribution is bimodal or has a heavy tail, the transcode is suspect. **Stop and investigate.**

### C.3 Check the FP8 → fp16 path
- The card claims all 390 FP8 tensors were converted to fp16 **bit-exactly**.
- Take one attention tensor from the source (FP8 e4m3 with e8m0 block scales) and one from the INT4 checkpoint (fp16).
- Dequantize the FP8 source and compare to the fp16 tensor.
- **Expected:** exact match, zero error. FP8 e4m3 values are discrete and representable in fp16, so this conversion should be lossless.
- If it is not exact, the dense path is corrupted and that alone explains incoherent output.

### C.4 Check for dead experts
- For every expert, compute the fraction of weights that are exactly zero after dequantization.
- The `BlivionIaG` card reports 10.1% dead experts in its own conversion. Check whether `yiminyuan` has a similar pathology.
- A dead expert is not necessarily a bug (it may reflect the source), but a *large* number of dead experts concentrated in specific layers is a red flag.
- Compare the dead-expert map against the source. If the source has live experts where the INT4 checkpoint has dead ones, the conversion destroyed them.

### C.5 Deliverable
A fidelity report: SNR per tensor, relative L2, dead-expert map, FP8 path exactness. This tells you whether the checkpoint is trustworthy.

---

## 4. Stage D — Your Repacked Format Verification (CPU only)

**Purpose:** Prove your byte-identical repack is lossless with respect to the community checkpoint.

### D.1 Round-trip test
- Take the INT4 checkpoint tensor from C.1.
- Run it through your repacking pipeline: split dense backbone from experts, write the expert index, write the custom byte layout.
- Read it back through your engine's loader.
- Dequantize using your engine's dequantization path.
- Compare against C.1 element-by-element.
- **Expected:** bit-exact. Zero difference. This is a pure format transformation; there is no numerical operation, so any difference is a bug in your packing or unpacking.

### D.2 Nibble order test
This is the highest-probability location of your current bug.
- Construct a synthetic tensor with known values: `[0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]`.
- Pack it with your writer, unpack it with your reader.
- Verify the round trip.
- Then verify the **byte layout** explicitly: which nibble holds the even-indexed element, which holds the odd-indexed element.
- Test both conventions and confirm which one your writer uses. A mismatch here swaps adjacent elements and produces plausible-looking but wrong output.

### D.3 Scale layout test
- Verify the scale tensor's shape, dtype, and ordering match what your kernel expects.
- Verify the group-to-scale mapping: element `i` belongs to group `i // 32`, and the scale index is `i // 32`.
- A transposed or offset scale array produces a smooth, subtle corruption that is easy to mistake for a kernel bug.

### D.4 Expert index test
- For every expert, use your index file to seek to its position, read it, and dequantize it.
- Compare against the same expert read sequentially from the community checkpoint.
- **Expected:** bit-exact for all 256 experts × 43 layers.
- This validates the index file, which is the part of your format that has no analogue in the source and therefore no reference to check against.

### D.5 Deliverable
A pass/fail on each of D.1 through D.4. If all pass, your format layer is correct and any remaining incoherence is in the kernels.

---

## 5. Stage E — Kernel Numerical Verification (GPU)

**Purpose:** Verify each kernel against a CPU reference, one at a time.

### E.1 Method
For each kernel, implement a naive CPU reference in fp64. Run the GPU kernel on the same input. Compare with a tolerance appropriate to the operation. Do not test kernels in combination until each passes alone.

### E.2 Order of testing
Test in dependency order, because a failure in an early kernel contaminates everything downstream:

1. **Dequantization kernel.** INT4 → fp16/bf16. Compare against the CPU dequantization from D.1. **Tolerance: exact.**
2. **RMSNorm.** Compare against fp64 reference. **Tolerance: ~1e-3 relative.**
3. **RoPE.** Test forward and inverse separately. The inverse RoPE on the last 64 dimensions of attention output is a common source of silent corruption. **Tolerance: ~1e-3 relative.**
4. **QKV projection.** Dense matmul. **Tolerance: ~1e-2 relative** (accumulation order differs).
5. **Attention scores and softmax.** Verify the attention sink logit is included. **Tolerance: ~1e-3.**
6. **Sinkhorn-Knopp.** Verify 20 iterations and that the output is doubly stochastic to within tolerance. **Tolerance: row and column sums within 1e-4 of 1.0.**
7. **Compressor.** Verify against reference.
8. **Indexer.** Verify INT8 path against an fp32 reference. **Tolerance: ~1e-2** (INT8 introduces quantization error, but the ranking should be preserved).
9. **MoE routing.** Verify top-k selection matches the reference exactly. A tie-breaking difference changes which experts fire.
10. **Expert matmul.** Verify against fp64 reference.
11. **Shared expert path.** Verify separately from routed experts.
12. **Output projection and `lm_head`.** Verify logits against reference.

### E.3 The end-to-end numerical test
Once every kernel passes individually, run a **single forward pass on a short prompt** and compare the final logits against a reference implementation (vLLM with the patched RDNA2 build, or a CPU reference if you have the patience). Compare the top-1 token and the top-5 logit values.

If the top-1 token matches but the logits differ slightly, you have a precision issue, not a logic issue. If the top-1 token differs, you have a logic issue, and you should bisect by comparing hidden states layer by layer.

### E.4 Layer-by-layer bisection
If the end-to-end test fails, compare the hidden state after each layer against the reference. The first layer where they diverge is where the bug is. This is the fastest way to localize a logic error and it is worth building the instrumentation for it before you need it.

---

## 6. Stage F — Streaming and Pool Correctness

**Purpose:** Verify the three-tier memory system does not corrupt data in transit.

### F.1 Expert integrity across pools
- Load an expert into VRAM, dequantize, record the result.
- Evict it, reload it from host RAM, dequantize, compare. **Expected: bit-exact.**
- Evict it, reload it from SSD, dequantize, compare. **Expected: bit-exact.**

### F.2 Concurrent streaming
- Run a forward pass while experts are being streamed in.
- Verify no expert is read while partially written.
- Verify the index file's positions remain valid under concurrent access.

### F.3 Deliverable
Confirmation that the memory hierarchy is transparent to the numerics.

---

## 7. Failure Triage Table

| Symptom | Most likely cause | Stage to check |
|---|---|---|
| Output is fluent but semantically wrong | Wrong nibble order, or scale offset | D.2, D.3 |
| Output is repetitive or degenerate | Missing attention sink, or Sinkhorn not converging | E.2 steps 5-6 |
| Output is coherent for a few tokens then collapses | KV cache corruption, or inverse RoPE error | E.2 step 3, F.1 |
| Output is random tokens | Wrong dequantization table, or wrong layer count | B.1, A.1 |
| Output is correct on short prompts, wrong on long | Indexer or compressor bug | E.2 steps 7-8 |
| Output differs only in rare tokens | Precision issue, not logic | E.3 |

---

## 8. Execution Order Summary

1. **Stage A** — structural audit. No compute. Do this first.
2. **Stage B** — source ground truth. CPU. Establishes the reference.
3. **Stage C** — community checkpoint fidelity. CPU. Tells you if the checkpoint is trustworthy.
4. **Stage D** — your repack round-trip. CPU. Tells you if your format is correct.
5. **Stage E** — kernel verification. GPU. One kernel at a time, in dependency order.
6. **Stage F** — streaming integrity. GPU. Last, because it depends on everything else being correct.

**Do not skip to Stage E.** The temptation is to start debugging kernels because that is where the interesting work is. But Stages B through D are cheap, they run on CPU, and they eliminate the two most likely causes of your current incoherence before you write a single line of GPU code.
