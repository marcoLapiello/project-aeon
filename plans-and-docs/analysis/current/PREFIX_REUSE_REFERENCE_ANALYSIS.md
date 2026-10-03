# Prefix reuse reference analysis — what to adopt for the session seam

Date `2026-10-03`. Question: prefix reuse is the engine-side capability that makes a live conversation stop paying the full-prompt cost every turn. Which reference implements **that**, on this model, closely enough to guide us? Method: read of the three references that carry session-prefix machinery — colibri, llama.cpp and ds4 — for the mechanism, the reuse check, and the reply-boundary behaviour. No Aeon code changed.

**Reference checkouts:** `/home/marcolap/aeon-references/colibri` (default branch), `9cf3bf2` at `/home/marcolap/aeon-references/llama.cpp`, `/home/marcolap/aeon-references/ds4` (default branch). Re-pull before trusting a specific revision.

**Verdict.** [colibri's `kv_prefix`](../../../../aeon-references/colibri/c/kv_prefix.h) is the primary reference: it is a **DeepSeek-V4**, **tiered-hardware** implementation of exactly this feature, self-contained, with the design rationale written into the header and measured costs. llama.cpp and ds4 are secondary — both implement the same idea, and each owns a refinement we may want, but neither is DSV4 on tiered consumer hardware. Where this document and the [server reference analysis](SERVER_REFERENCE_ANALYSIS.md) overlap, that one covers G5's transport; this one covers the engine-side seam behind it.

---

## 1. The mechanism, as colibri writes it

The whole idea is one small record. `kv_prefix` holds `int *fed` — *the token ids the current state was built from, in position order* — plus `len`, `cap`, and a `tainted` flag. Reuse is: if the new prompt's ids begin with exactly the recorded ids, skip the reset and prefill only the tail.

| Concern | File | Symbol | What we take |
| :--- | :--- | :--- | :--- |
| **The record + policy** | [kv_prefix.h](../../../../aeon-references/colibri/c/kv_prefix.h) | `kv_prefix`, `kv_prefix_record`, `kv_prefix_reuse`, `kv_prefix_clear`, `kv_prefix_grow`, `kv_prefix_taint` | The complete mechanism in one header. `fed` is the single description of what the state covers; `reuse` returns the coverage or `0`. |
| **Where it is fed** | [deepseek_v4.c](../../../../aeon-references/colibri/c/deepseek_v4.c) | `kv_prefix_record` at 13119/13161/13371/13394, `kv_prefix_reuse` at 13003 | Record **at the feed site**, per position, and nowhere else. |
| **Where reuse is decided** | [deepseek_v4.c](../../../../aeon-references/colibri/c/deepseek_v4.c) | the prefill entry, `int reuse = kv_prefix_reuse(...)` at 13003 | Set `reuse` as the prefill's start position; the tail is prefilled at that absolute start. |
| **Session ownership** | [deepseek_v4_internal.h](../../../../aeon-references/colibri/c/deepseek_v4_internal.h) | `kv_prefix fed;` in `ColiV4Session` (963) | The record is part of the session, alongside the state it describes. |

Three design rules in the header are worth copying verbatim into our plan:

- **A record, not a counter.** The reusable length must not be derived from the caller's bookkeeping (`prompt_count + generated - 1`) — "that invariant differs per engine … and getting it wrong does not crash: it silently answers from a state that belongs to a different conversation." The ids are recorded **where they are fed**, and the record is the only description of the state anyone consults. This is exactly the "the session aggregate must carry the token prefix" point from [SESSION_STATE_AND_SWAP_ANALYSIS.md](SESSION_STATE_AND_SWAP_ANALYSIS.md) §3.
- **`kv_prefix_grow` must preserve the record.** When the state buffers grow (a longer prompt), the record has to grow with them, or reuse can never fire in the one case it exists for — a conversation whose prompt lengthens every turn.
- **Failure is not an error.** If `kv_prefix_alloc` fails, the record is empty, `kv_prefix_reuse` returns `0`, and "every request prefills in full … this is an optimisation and must never be the reason a turn fails." That is the R6 discipline, applied to the reuse path itself.

---

## 2. Why the check is *strict prefix* and not longest-common-prefix

This is the DSV4-specific part, and it is the reason colibri is a better guide than a generic server. The state here is not a plain per-position KV matrix that can be truncated: the sliding window is a **ring**, and the compressor carries **recurrent** `kv_state`/`score_state` rather than addressable rows. So the engine **cannot rewind** — there is no "delete positions ≥ k" operation, which is precisely the four-piece, position-addressed state our session analysis describes (local ring, compressed entries, compressor partial, indexer). From colibri's own comment at the reuse decision:

> the sliding window is a ring and the compressor carries recurrent kv_state/score_state rather than per-position rows. What it *can* do is keep going. So the reusable case is the exact one a conversation produces: turn N+1's prompt begins with every id turn N fed, prompt and reply alike, and only the tail is new.

Two consequences for us — both already latent in our requirements, now with a reference behind them:

- **`kv_prefix_reuse` requires at least one new token** (`len < n`). A prompt equal to, or shorter than, the recorded sequence would need a **rewind**, "which nothing here can do — and prefilling zero tokens leaves the caller with no final hidden state to sample from." Our seam must refuse reuse unless there is a non-empty tail.
- **All-or-nothing, not longest-common-prefix.** colibri reuses the *entire* recorded prefix or none of it: a divergent prompt "falls back to a full reset and prefill". **This is the simpler correct policy for our state**, and it is the one to adopt. Partial reuse would require truncating the compressed/compressor-partial/indexer state at the divergence point — the hard operatation we do not have. (llama.cpp takes the longest-common-prefix route, §4, because `llama_memory` supports `seq_rm`; we do not, and do not need to.)

**Design decision to record:** our R3 currently says a rejected reuse "replays from the point of divergence". The correct and cheaper implementation is **replay from the start** — correctness is preserved (a cold run produces the same logits, as colibri notes: "positions stay absolute and the tail is prefilled at start=reuse, so the logits are the ones a cold run would have produced"). R3's *intent* (correctness survives a rejected reuse) is met; only its mechanism wording is stricter than our state can support. The plan should settle this, and R3 may be relaxed to "replays" without naming the point.

---

## 3. The finding that decides the agent case: the reply boundary

The single most valuable thing in colibri for our requirements is not in `kv_prefix.h` — it is the note above the checkpoint struct, and it concerns **exactly the agent workload R1 names as a target**:

> PLAN captures are the discovered system prefix. **PROMPT-END captures are taken after every prefill at prompt_count: an agent's next turn re-renders the assistant reply (tool calls, stripped reasoning), so strict-prefix session reuse fails at the reply boundary** and without this snapshot the whole conversation re-prefilled (measured: opencode turn 2 = **677 s for 66 new tokens**). Eviction takes the LRU prompt-end slot first so the system prefix survives a long session.

Read that against our R2/R3. Strict-prefix reuse assumes turn N+1's prompt begins with **every id turn N fed, prompt and reply alike**. For a **plain chat**, that holds: the reply text is re-sent unchanged and re-tokenizes to the same ids. For an **agent loop**, it does not: the client re-renders the assistant turn — tool calls reformatted, reasoning stripped — so the token sequence diverges **at the reply boundary**, and strict prefix fails every turn. colibri's answer is a **snapshot taken at prompt-end** (state covering up to the end of the *user* prompt, before the assistant reply), restored when the re-rendered tail diverges.

This is a decision the plan must make, and it is genuinely open:

| Workload | Strict prefix suffices? | Reference |
| :--- | :--- | :--- |
| **Plain multi-turn chat** | **Yes** — reply re-sent verbatim | colibri `kv_prefix`; measured reuse `61 s vs 320 s` on DSV4 |
| **Agent loop** (re-rendered reply) | **No** — diverges at the reply boundary | colibri prompt-end capture / `v4_ckpt_restore` |

Our requirements name both workloads. So the plan should either (a) scope near-term reuse to the plain-chat case and accept a full re-prefill after a tool turn, or (b) include a **single in-memory prompt-end snapshot** so agent turns reuse too. Option (b) is more work — a per-layer snapshot each turn, ~`250 MB` for a 2k window per colibri — and it is the mechanism we already have (`V4LayerStateSnapshot`/`restore_state`), so it is a wiring cost, not new correctness. This is the "session storage" boundary approached from the near side; worth stating explicitly rather than discovering in use.

**Note on the two measurements** to quote in the plan: colibri reports a plain reuse of `82%` of a prompt taking `61 s` instead of `320 s`, and the agent failure case at `677 s` for `66` new tokens without a prompt-end snapshot. Both are DSV4 on the sibling tiered engine, so they are the closest available substitute for a number we have not measured yet.

---

## 4. Reuse invalidation — colibri's `taint` is our R4

`kv_prefix_taint` marks state that consumed something **token ids cannot describe** (colibri's instance: audio frames that all carry the same id while the mel payload differs), and a tainted record is never reused. This is exactly our R4 in a shipping implementation: an input that changes the computation but is invisible in the token sequence must invalidate reuse. Our instances are thinking mode, reasoning effort, active tool set and response format. The mechanism to copy is the shape — **a single `tainted` flag that forces `reuse = 0`** — and the discipline of setting it *at the point the non-token input is consumed*, not by a later comparison.

---

## 5. Secondary references

Each implements the same mechanism; each owns one thing colibri does not.

| Reference | File | Symbol | What it adds |
| :--- | :--- | :--- | :--- |
| **llama.cpp** — longest-common-prefix reuse | [server-common.cpp](../../../../aeon-references/llama.cpp/tools/server/server-common.cpp) | `server_tokens::get_common_prefix` (680) | The LCP primitive, straight equality over ids. The variant of the check we are *not* adopting, but the clearest statement of it. |
| **llama.cpp** — reuse in the slot loop | [server-context.cpp](../../../../aeon-references/llama.cpp/tools/server/server-context.cpp) | `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)` (3219) | "Reuse any previously computed tokens common with the new prompt" — reuse expressed as a **start position**, which is exactly how our `forward_window(token_ids, start_position, …)` already works. |
| **llama.cpp** — reuse only when asked | [server-context.cpp](../../../../aeon-references/llama.cpp/tools/server/server-context.cpp) | `if (slot.task->params.cache_prompt)` (3216) | Reuse is a per-request policy, not automatic. Worth mirroring as a seam parameter. |
| **ds4** — verified byte-prefix match | [ds4_kvstore.c](../../../../aeon-references/ds4/ds4_kvstore.c) | `ds4_kvstore_byte_prefix_match` (`memcmp`), `ds4_kvstore_find_text_prefix` | The **verify, never assume** shape of R3, at byte granularity, plus a hash check of the cached text before trusting the payload. |
| **ds4** — token history vs re-tokenized text | [ds4_kvstore.c](../../../../aeon-references/ds4/ds4_kvstore.c) | `ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix` | The exact remedy for the reply-boundary problem: keep the **exact token history** the state was built from, and tokenize only the text suffix of the new prompt. Confirms §3's divergence is real and gives the reconstitution rule. |
| **ds4** — live continuation | [ds4_server.c](../../../../aeon-references/ds4/ds4_server.c) | `anthropic_prepare_live_continuation`, `anthropic_live_has_call_id` | Detects a tool-result tail and continues the live session rather than re-prefilling — the agent case handled at the transport. |
| **ds4** — live continuation | [ds4_server.c](../../../../aeon-references/ds4/ds4_server.c) | `anthropic_prepare_live_continuation`, `anthropic_live_has_call_id` | Detects a tool-result tail and continues the live session rather than re-prefilling — the agent case handled at the transport. |

The two llama.cpp rows together are the important secondary idea: **reuse is a start position**, and it is opt-in per request. That maps one-to-one onto our existing `forward_window` signature and the neutrality of the seam.

---

## 6. Deferred reference — snapshot restore and cross-session prefixes

Recorded so it is not lost, not so it is started. colibri's checkpoint machinery is the reference for the capabilities we deferred, and it is closer than the servers' because it is DSV4 and snapshots the *same four-piece state*.

| Deferred capability | File | Symbol |
| :--- | :--- | :--- |
| Snapshot restore by longest-prefix match | [deepseek_v4.c](../../../../aeon-references/colibri/c/deepseek_v4.c) | `v4_ckpt_restore` (system-prefix restore when strict reuse fails) |
| Snapshot slots, kinds and eviction | [deepseek_v4.c](../../../../aeon-references/colibri/c/deepseek_v4.c) | `V4PrefixCkpt` (`ids`, `snapshots`, `kind` = plan / prompt-end), `v4_ckpt_slots`, `V4_CKPT_MAX_SLOTS` |
| Disk persistence of snapshots | [deepseek_v4.c](../../../../aeon-references/colibri/c/deepseek_v4.c) | `v4_ckpt_disk_init` / `v4_ckpt_disk_write` / `v4_ckpt_disk_path` |
| Restore-by-identity (llama.cpp) | [server-task.cpp](../../../../aeon-references/llama.cpp/tools/server/server-task.cpp) | `server_prompt_cache::load` — `get_common_prefix` + the `f_keep`/`f_sim` heuristic ("don't trash large prompts", `f_keep < 0.25` rejected) |
| Disk store with eviction reasons (ds4) | [ds4_kvstore.h](../../../../aeon-references/ds4/ds4_kvstore.h) | `ds4_kvstore_find_text_prefix`, `ds4_kvstore_try_load_text`, `DS4_KVSTORE_REASON_{COLD,CONTINUED,EVICT,SHUTDOWN}` |

The distinction that matters: **strict-prefix reuse is the near-term feature; snapshot restore is the matcher/session-storage tier.** colibri implements both in one file, and the boundary between them is the reply-boundary divergence of §3 — which is the same boundary between our in-scope prefix reuse and our deferred session storage.

---

## 7. How this maps to the requirements

| Requirement | Backed by |
| :--- | :--- |
| R1 reuse, turn N+1 computes only its addition | `kv_prefix_record`/`kv_prefix_reuse`; `n_past = get_common_prefix(...)`; reuse as a start position |
| R2 continuation == from-scratch, at token granularity | "positions stay absolute and the tail is prefilled at start=reuse, so the logits are the ones a cold run would have produced" |
| R3 verified, never assumed; correct on rejection | `kv_prefix_reuse`'s exact `memcmp`; ds4 `byte_prefix_match` + hash; on failure, full reset (§2) |
| R4 non-token inputs invalidate reuse | `kv_prefix_taint` — one flag, set where the input is consumed |
| R5 state bounded by context | `kv_prefix_grow` — growing buffers must preserve the record, and the record is sized to the state |
| R6 inspectable, failure never fatal | `kv_prefix_alloc` failure disables reuse, never fails the turn |
| Agent/tool workload (R1 target) | prompt-end capture (§3) — **open decision** |

---

## 8. One-line summary

colibri's [`kv_prefix`](../../../../aeon-references/colibri/c/kv_prefix.h) is the reference to build from: **record the ids where they are fed, reuse only when the new prompt begins with exactly that sequence, never rewind, and taint state fed anything the ids cannot describe** — all true of our four-piece DSV4 state, with the payoff measured on this model (`61 s vs 320 s`). The plan must decide one thing colibri exposes: **agent turns re-render the reply, so strict-prefix reuse fails at the reply boundary**, and colibri answers that with a prompt-end snapshot (`v4_ckpt_restore`) — the near edge of the session storage we deferred. llama.cpp contributes the secondary ideas that reuse is a **start position** and is **opt-in per request**; ds4 contributes the **exact-token-history + text-suffix reconstitution** and the byte-level verification discipline.
