# Prefix Reuse — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** the engine-side capability that makes a live conversation stop paying the full-prompt cost on every turn — **prefix reuse**. It is an engine concern, validated at the engine level.

**See also:** [Prefix reuse reference analysis](../../analysis/current/PREFIX_REUSE_REFERENCE_ANALYSIS.md) — the DSV4 reference for this seam, including the reply-boundary decision that shapes R1–R4. [Session state and swap analysis](../../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) — the deferred session-storage capability this sits in front of.

---

## 1. What we are building

The engine stops treating every reply as a fresh computation. A **conversation is a live session** — the model state produced by all preceding turns stays resident, and each new turn computes only what the turn added.

The workload is a single, explicit one: **one live conversation**, whether it is a human chatting over many turns or an agent extending its own context with tool results.

---

## 2. Requirements

**R1 — A turn computes only its addition.**
When turn N+1 is built on turn N, the prefill cost is proportional to the tokens turn N+1 *added*, not to the whole conversation. Observable: the time to first token on a continuation turn is governed by the appended length, not the accumulated length.

**R2 — A continuation is exactly a from-scratch run, at token granularity.**
A conversation reached turn-by-turn produces the **same generated token ids** as the same conversation rendered and prefilled in one shot, under greedy decoding (or a fixed seed). This is a correctness requirement, not a performance one. It is deliberately stated at token granularity, not at bit granularity: the from-scratch run processes the prompt through the prefill chunk path while a continuation processes earlier turns through the decode path, and those are different kernels with different accumulation orders — bit-equality is not a property correct code produces. Bit-equality is kept as a **diagnostic** printed alongside the comparison, never a pass/fail condition. The comparison fixes sampling and compares the same rendered token sequence, so it measures prefix reuse and not the tokenizer's detokenize/retokenize round-trip.

**R3 — Reuse is verified, never assumed.**
Before reusing resident state the engine establishes that the new prompt genuinely extends the resident one. If it does not — a re-rendered template shifts earlier tokens, the user edited an earlier message, the system prompt changed — the engine does **not** fail and does **not** serve stale state. It replays and produces a correct result. Correctness survives a rejected reuse. Reuse also requires at least one new token: a prefill of zero tokens leaves no logits to sample from, so an equal-or-shorter prompt replays rather than reuses.

**R4 — Everything that changes the computation invalidates reuse.**
The non-token inputs that alter the graph — thinking mode, `drop_thinking`, reasoning effort, active tool set, response format — are part of what decides whether the resident state is still valid for the new turn. A change in any of them is treated exactly like a change in the tokens.

**R5 — The state stays bounded by the configured context.**
Reuse does not grant unbounded context. The resident state lives inside the declared context capacity, and the budget accounts for it against the expert pool. Exceeding capacity is a stated, up-front refusal. A long conversation costs the Hot expert pool, and that cost is known and configured, not discovered at runtime.

**R6 — Deterministic and inspectable.**
For a fixed conversation, sampling setting, and seed, the run is reproducible. Failures and refusals are visible and diagnosable, not silent.

---

## 3. In scope

- Incremental prefill for **one live conversation**, at the engine level, validated before any transport exists.
- The validity check for reuse, and the replay fallback that keeps R2 true when it fails.
- The non-token inputs as part of session identity (R4).
- The accommodation of the resident state inside the memory budget, with the resulting context/pool trade made explicit (R5).
- The conversation as the engine's unit of work.

---

## 4. Out of scope

- **Session storage / persistence** — writing session state to SSD and reloading it. On restart, or on switching conversations, the answer is a fresh prefill. This is a separate capability with its own design, deliberately deferred.
- **Prefix matching across sessions** — radix trees, block tables, cache keys, eviction, sharing a system prompt or forking a child session from a parent prefix. Deferred; with a single live session there is nothing to match.
- **Continuous batching** — multiple sequences stepping together through the graph. The engine is single-sequence by design.
- **Multiple concurrent sessions** — more than one live conversation resident at once.
- **Any change to the model, its graph, or the expert tier's strategy.** This work is additive around the existing forward pass.
- **Speculative decoding / MTP**, multi-GPU parallelism, KV-precision changes, tool *execution* beyond prompt encoding.

---

## 5. What this buys, and what it honestly does not

It makes **one long conversation usable**: a multi-turn chat or an agent loop stops paying the full-prompt cost on every turn, which is the difference between an interactive product and a ~20-second pause per message.

It does **not** survive a restart without a fresh prefill, and it does **not** switch between conversations without one — those are the things session storage would add, knowingly left for later.

---

## 6. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Session storage / persistence (`O_DIRECT` store, identity, eviction) | a separate capability with its own design; the near-term workload is one live conversation, which never leaves VRAM | prefix reuse is green and concurrent conversations or restart recovery are actually needed |
| Prefix **matcher** (block table, cache key, radix search, eviction) | its parameters are measurements of an assembled graph; a single live session needs no key and nothing to match | session storage exists and a fork / shared-system-prompt workload exists |
