# Routing Profile and Frequency-Informed Placement Study

*Status: **Phase 1 complete** (reuse-distance + Belady-OPT, see [Expert Streaming Execution Plan §6.10](./../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) and ledger M35/M36). **Phase 2 scoped below.** The aggregation library exists and the driver draft needs adjustment to the rebuilt runtime before use.*

## 1. Purpose

Produce a reliable per-layer activation ranking for the 256 routed experts in the 43-layer DeepSeek-V4 model. The first result of this study is data, not a new cache policy:

- for every layer, rank all 256 experts by observed selection frequency;
- preserve the measured probability for every expert, including experts with zero observations;
- provide the evidence needed to decide whether frequency-informed Hot/Warm placement is worthwhile.

This study must not change normal inference behavior, the default expert placement policy, or the established benchmark and regression tests.

## 2. Measurement boundary

The profiler must observe the routing result once the six selected expert IDs are known on the host. The graph exposes this seam as `V4RoutedExpertExecutor::on_routing_ready(layer_id, position, ids, weights)` in `src/architecture/deepseek_v4/core/v4_expert_executor.hpp`, which runs after the routed ids are known and before any device work for the MoE; the profiler counts those IDs rather than adding a second router path or collecting router logits.

The smallest record needed for aggregation is:

```text
prompt_id
position
phase       # prefill or decode
layer_id
expert_id[6]
```

The first implementation does not collect router logits, routing weights, entropy, Gini coefficients, transition matrices, or speculative candidate sets. Those are secondary studies and must not expand the initial measurement scope.

## 3. Workload requirements

A single synthetic prompt is not sufficient. The profile must use a corpus of many real, tokenized prompts with enough variety to avoid making the ranking specific to one short trace.

The native runtime contains its own tokenizer and prompt encoder, so the corpus can be tokenized in-runtime. The workflow is:

1. Prepare a text corpus.
2. Format and tokenize each prompt with the native encoder and tokenizer.
3. Store the tokenized prompts and profiling parameters in a structured JSONL input file.
4. Run the native profiler through the full 43-layer model.

Example input row:

```json
{"id":"prompt_0001","tokens":[101,2045,778,19],"max_new_tokens":64}
```

Each prompt must use the complete 43-layer model. Layer-by-layer or shortened-model runs are not valid for the ranking because each layer's hidden state depends on the preceding layers and the full attention context.

The profiler must label prefill and decode separately. The initial ranking outputs should include:

- `prefill`: prompt processing tokens;
- `decode`: autoregressively generated tokens;
- optionally `combined`: an explicit sum of the two distributions, never an implicit mixture.

The corpus must be divided into a profile set and a held-out validation set. The profile set builds the ranking; the held-out set later measures whether that ranking generalizes.

## 4. Aggregation

For each phase, layer, and expert, maintain a counter:

```text
selection_count[phase][layer_id][expert_id]
```

Every processed token contributes six selections per layer. For layer `l` and expert `e`, calculate:

$$
p_l(e) = \frac{\text{selection_count}_l(e)}{\sum_{j=0}^{255} \text{selection_count}_l(j)}
$$

Sort the 256 experts in descending order of `p_l(e)`. Ties must have a deterministic secondary order, such as ascending expert ID.

The required ranking artifact is one row for every phase, layer, rank, and expert:

```text
phase,layer_id,rank,expert_id,selection_count,total_selections,probability,cumulative_probability
```

The first study only requires the ranking and probability data. Cumulative probability is included because it directly answers how much observed routing traffic is covered by the first `k` experts for a layer and is inexpensive to derive from the ranking.

## 5. Durable output and resumability

The profiler must write structured files instead of requiring terminal-log extraction. The proposed output directory is:

```text
routing-profile/<run_id>/
    metadata.json
    state.bin
    counts.csv
    ranking.csv
    summary.csv
    progress.json
```

`metadata.json` must identify at least:

- Aeon git commit;
- model/config identity;
- model and tokenizer/corpus identifiers or hashes;
- layer count, expert count, and top-k;
- profile input path and prompt count;
- completed and failed prompt IDs;
- profiler version and run parameters.

`state.bin` is the atomic canonical aggregate state. It stores the counters, completed prompt signatures, and failed prompt records. `counts.csv`, `ranking.csv`, `summary.csv`, and `progress.json` are structured views regenerated from that state after each completed prompt. `ranking.csv` is derived from the aggregate counts; `summary.csv` is the compact human-readable report; `progress.json` mirrors the completed prompt IDs used for resume decisions.

`summary.csv` contains one row per phase and layer. It reports total selections, the top expert and its probability, cumulative coverage at the top 4, 8, 12, 16, and 32 experts, and the IDs of the top 12 experts. For the complete 43-layer model it therefore contains 86 data rows instead of the 22,016 rows in the full ranking.

To rebuild this compact view without loading the model or rerunning inference, invoke `profile_routing --output-dir <directory> --regenerate-summary`. This maintenance mode reads only the validated `state.bin`; it does not bypass compatibility checks for normal incremental profiling runs.

Each update must be crash-tolerant: write a temporary file in the same directory, flush and close it, then atomically rename it to the target path. A partially written ranking must never be treated as a completed result.

The profiler should process one prompt at a time and checkpoint after each successful prompt. A failure for one prompt must be recorded and must not silently invalidate the rest of the run.

### Incremental multi-day runs

The same profiling run must support adding prompts over time. For example, a run may process four prompts on the first day, ten additional prompts on the next day, and twenty more later. Each new prompt contributes to the existing cumulative counters and causes the ranking to be regenerated.

Prompt IDs must be stable and unique within a run. Before processing a prompt, the profiler checks `progress.json`; a completed prompt is skipped or rejected and must never be counted twice. A prompt that fails is not marked complete and may be retried explicitly.

The run stores a compatibility fingerprint in `metadata.json`. New prompts may be added to an existing run only when the model revision, tokenizer/corpus identity, layer count, expert count, top-k, phase rules, and generation settings match that fingerprint. If any of these inputs change, the profiler starts a new run directory rather than mixing incompatible observations.

The aggregate state is the source of truth across invocations. The ranking is a derived snapshot and must be rebuilt from that state after every successful prompt, so the current ranking is always available even when data collection is paused between batches.

## 6. Implementation shape

Add an optional routing observer behind the `on_routing_ready` hook. It is disabled by default and has no effect on normal execution when disabled. It receives the layer ID, position, phase, and six expert IDs.

Add a dedicated executable, named `profile_routing`, with responsibilities for:

- reading the tokenized JSONL corpus;
- selecting profile or validation prompts;
- configuring the full 43-layer pipeline;
- enabling the routing counter;
- running the existing generation path;
- writing checkpoints and aggregate files;
- reporting only concise progress and output paths to the terminal.

The profiler must not duplicate model execution logic. It should call the same generation path as the engine.

## 7. Validation stages

### Stage 1: instrumentation smoke test

Use the existing short synthetic input only to verify the instrumentation:

- output tokens remain unchanged;
- all 43 layers produce records;
- six selections are counted per layer and processed token;
- all 256 expert rows are emitted;
- output files can be read after interruption and resumed.

### Stage 2: small corpus pilot

Run a small set of real tokenized prompts through the complete model. Confirm corpus parsing, checkpointing, phase labels, and expected count totals before starting a long run.

### Stage 3: profile corpus

Run the full profile corpus and produce the 43 x 256 rankings for prefill and decode. Keep the held-out corpus separate and do not use it to build the ranking.

### Stage 4: later placement evaluation

Only after the ranking exists, replay held-out traces or run the runtime with an experimental placement policy. Compare static frequency-ranked placement with the current dynamic LRU policy. Do not modify the default policy as part of the profiling study.

## 8. Benchmark decision

`profile_routing` is the dedicated driver for multi-prompt data collection. It handles corpus input, full-model repeated runs, phase labeling, persistent output, checkpointing, and resumption.

The profiler must reuse the engine's generation path and model initialization; it must not create a simplified or partial profiling model.

## 9. Definition of done for the first study phase

This phase is complete when:

1. The default inference and existing regression tests remain behaviorally unchanged.
2. `profile_routing` can process tokenized prompts through all 43 layers.
3. A run produces durable metadata, counts, rankings, a compact summary, and progress files without terminal extraction.
4. An interrupted run resumes from the last completed prompt.
5. The output contains exactly 256 ranked expert rows for every measured phase and layer.
6. The profile and held-out corpora are independently identifiable.
7. Additional compatible prompts can be added on later invocations without double-counting or losing previous observations.
8. Incompatible model or profiling settings are rejected or placed in a new run directory.
9. The profile results are sufficient to choose the next step: frequency-informed placement, a different locality study, or abandoning static frequency placement.

Placement changes, speculative prefetch, entropy analysis, and other secondary metrics are explicitly outside this first implementation phase.

## 10. Phase 2 — is the headroom real, and is it capturable?

*Added 2026-09-21, after Phase 1. Phase 1 answered the recency question and left exactly this one open.*

**What Phase 1 established.** Ideal-LRU equals the measured Hot hit rate (`60.5%` vs `60.6%` at `779` slots), so the current recency policy is at its ceiling — **LRU is not thrashing**. But Belady-OPT (an oracle that sees the future) reaches `77.4%` at the same capacity, so a **non-recency policy has up to `+16.9` points of headroom**. That is an *upper bound*: it says the prize exists, not that a real policy captures it.

**What Phase 2 must decide.** Whether a **practical, online** frequency-aware policy captures a useful fraction of that gap on *unseen* workloads. Two gates, in order:

| Gate | Question | Instrument | Decision if it fails |
| :--- | :--- | :--- | :--- |
| **P2-a: skew** | Within each layer, does a small top-k of experts carry most of that layer's selections? | `RoutingCounter` per-layer counts; coverage at top-4/8/16/32 (already computed by `RoutingProfile`) | If routing is flat per layer, no frequency policy helps — abandon frequency placement, keep coverage/capacity as the only lever |
| **P2-b: generalization** | Does a ranking built on corpus A predict corpus B? | profile set vs held-out set; rank correlation + held-out coverage | If it does not generalize, a *pre-ranked static* placement is dead. Only an **online adaptive** policy (no pre-ranked list) remains, which is a larger build |

P2-a is cheap (the existing counters plus a small driver). P2-b is the gate that decides *which* design is even buildable, so it is not optional.

**The design intent this study is meant to inform — dynamic, not pinned.** The target is **not** a static placement that pins a fixed top-k per layer forever. That would be wrong twice: the popular experts differ by workload, and a pin has no adaptation. The target is a **dynamic eviction policy**: experts keep flowing between Hot, Warm and Cold; only the rule that chooses the victim changes — from *least recently used* to *least frequently used*. And not raw frequency: **frequency with decay**, so an expert that was hot early does not clog the pool forever. The registry already records exactly this signal — `ExpertCatalogEntry::moving_frequency` is an EMA (`f ← 0.9·f + 0.1` on each activation), currently recorded and never read (plan §5). A Phase-3 policy would read it.

**Also worth measuring here (secondary, cheap).** Per-layer coverage gives the aggregate view; a direct measure of the **temporal** structure — reuse-distance and OPT curves computed *per layer* rather than globally — would say which layers are cacheable at all. It is the same instrument as Phase 1, applied per layer.

**Out of scope for Phase 2.** Changing the default placement policy, speculative prefetch, entropy/transition analysis, and any router-logit collection. Phase 2 produces **data**, and its output is a decision about whether a Phase-3 policy is worth building.
