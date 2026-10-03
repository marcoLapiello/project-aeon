# Server and Prefix Reuse — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** the single write-up that fixes the goals, the requirements, and the in-scope / out-of-scope boundary for the near-term work: **prefix reuse** in the engine and the **server** built on top of it.

**See also:** [Server reference analysis](../../analysis/current/SERVER_REFERENCE_ANALYSIS.md) — the reference server whose transport shell G5 draws on, and where the reusable layer ends.

---

## 1. What we are building

**One capability and one product surface.**

The capability: the engine stops treating every reply as a fresh computation. A **conversation is a live session** — the model state produced by all preceding turns stays resident, and each new turn computes only what the turn added. This is **prefix reuse**.

The product surface: a **server** that lets a person or a program hold that one conversation over a network — send a message, receive the reply as it is generated — instead of invoking a CLI once per prompt.

Both serve a single, explicit workload: **one live conversation**, whether it is a human chatting over many turns or an agent extending its own context with tool results. That is the whole near-term target.

---

## 2. Requirements

### Prefix reuse

**R1 — A turn computes only its addition.**
When turn N+1 is built on turn N, the prefill cost is proportional to the tokens turn N+1 *added*, not to the whole conversation. Observable: the time to first token on a continuation turn is governed by the appended length, not the accumulated length.

**R2 — A continuation is exactly a from-scratch run, at token granularity.**
A conversation reached turn-by-turn produces the **same generated token ids** as the same conversation rendered and prefilled in one shot, under greedy decoding (or a fixed seed). This is a correctness requirement, not a performance one. It is deliberately stated at token granularity, not at bit granularity: the from-scratch run processes the prompt through the prefill chunk path while a continuation processes earlier turns through the decode path, and those are different kernels with different accumulation orders — bit-equality is not a property correct code produces. Bit-equality is kept as a **diagnostic** printed alongside the comparison, never a pass/fail condition. The comparison fixes sampling and compares the same rendered token sequence, so it measures prefix reuse and not the tokenizer's detokenize/retokenize round-trip.

**R3 — Reuse is verified, never assumed.**
Before reusing resident state the engine establishes that the new prompt genuinely extends the resident one. If it does not — a re-rendered template shifts earlier tokens, the user edited an earlier message, the system prompt changed — the engine does **not** fail and does **not** serve stale state. It replays from the point of divergence and produces a correct result. Correctness survives a rejected reuse.

**R4 — Everything that changes the computation invalidates reuse.**
The non-token inputs that alter the graph — thinking mode, reasoning effort, active tool set, response format — are part of what decides whether the resident state is still valid for the new turn. A change in any of them is treated exactly like a change in the tokens.

**R5 — The state stays bounded by the configured context.**
Reuse does not grant unbounded context. The resident state lives inside the declared context capacity, and the budget accounts for it against the expert pool. Exceeding capacity is a stated, up-front refusal. A long conversation costs the Hot expert pool, and that cost is known and configured, not discovered at runtime.

**R6 — Deterministic and inspectable.**
For a fixed conversation, sampling setting, and seed, the run is reproducible. Failures and refusals are visible and diagnosable, not silent.

### The server

**R7 — A conversational endpoint.**
The server accepts a conversation (messages plus the non-token inputs of R4) and returns the model's reply. This is the same conversation abstraction the engine already consumes — the server adds transport, not a second notion of "conversation."

**R8 — Streaming.**
Tokens reach the client as they are generated. The client sees the reply being produced, not a final blob after a multi-second wait. First-token latency matters as much as total latency.

**R9 — One conversation at a time, by design; a second one queues.**
Exactly one live session is resident. A second conversation that arrives while one is live is **queued**, not refused. The queue is a fairness and turn-taking policy: a queued conversation is a *pending fresh conversation*, not a saved session — since only one session is resident and there is no session storage yet, when its turn arrives it prefills from scratch. Nothing is silently interleaved, and a queued request never corrupts the live session. The queue is the seam a future multi-session scheduler would grow from; single-session is a deliberate design choice, not a limitation to be worked around in this campaign.

**R10 — Cancellation.**
A generation in flight can be stopped by the client. Cancellation leaves the server and the resident conversation in a well-defined state, and does not corrupt the session.

**R11 — Robustness.**
Malformed input, a request too long for the context, an interrupted connection, an internal failure — each produces a defined error response and leaves the server running and the session coherent. The server never crashes on bad input and never answers with a corrupt result.

**R12 — Operability.**
The operator can see: whether a session is live, what it currently holds, whether a reuse was accepted or replayed, and the last request's timings. Enough to answer "why was that slow" without attaching a debugger.

**R13 — The product installs as one piece.**
The end user obtains Aeon as a **single product** and starts serving with **one command** — the way `vllm serve` or `llama-server` work. The server is not a separately installed component with its own dependency set: it builds and ships with the engine. The engine's zero-runtime-dependency discipline (AGENTS.md rule 1) is a property of **the whole product the user installs**, not of the engine binary alone — a user must never have to assemble the pieces or satisfy a second dependency stack to get a working server.

**R14 — The server is model-agnostic.**
The server holds **no model, weight-format, or kernel knowledge** (G5 depends on no G3/G4 type). The engine presents a conversation and streams back **decoded text** plus neutral scalars (stop reason, timings, token counts); the server never tokenizes, detokenizes, or renders a chat template. This is what lets a future model architecture or a second transport be added without touching the other side, and it fixes the dependency direction: the engine never depends on the server. The same neutral seam is what the tests and the CLI use, so "one conversation" has exactly one definition in the codebase.

---

## 3. In scope

- Incremental prefill for **one live conversation**, at the engine level, validated before any transport exists.
- The validity check for reuse, and the replay-from-divergence fallback that keeps R2 true when it fails.
- The non-token inputs as part of session identity (R4).
- The accommodation of the resident state inside the memory budget, with the resulting context/pool trade made explicit (R5).
- A server that serializes access to the single session: request intake, queuing, streaming output, cancellation, defined errors, basic operability (R7–R12).
- The **neutral conversation seam** between the engine and the server — conversation in, decoded text streamed out, with neutral scalars — so the server carries no model knowledge and the engine never depends on the server (R14).
- A **single build and install** that produces the engine and the server as one product, launched with one command (R13).
- The conversation as **the** unit of work — the same unit the CLI, the tests, and the server all use.

---

## 4. Out of scope

- **Continuous batching** — multiple sequences stepping together through the graph. Explicitly excluded; the engine is single-sequence by design.
- **Multiple concurrent sessions** — more than one live conversation resident at once. A second conversation queues (R9); it does not run concurrently.
- **Session storage / persistence** — writing session state to SSD and reloading it. On restart, or on switching conversations, the answer is a fresh prefill. This is a separate capability with its own design, deliberately deferred.
- **Prefix matching across sessions** — radix trees, block tables, cache keys, eviction, sharing a system prompt or forking a child session from a parent prefix. Deferred; with a single live session there is nothing to match.
- **Any change to the model, its graph, or the expert tier's strategy.** This work is additive around the existing forward pass.
- **Speculative decoding / MTP**, multi-GPU parallelism, KV-precision changes, tool *execution* beyond prompt encoding.

---

## 5. What this buys, and what it honestly does not

It makes **one long conversation usable**: a multi-turn chat or an agent loop stops paying the full-prompt cost on every turn, which is the difference between an interactive product and a ~20-second pause per message. It gives that capability a real interface, usable by a human or a program.

It also gives the result a **real product shape**: one install, one command to serve, the same way a user expects of `vllm` or `llama.cpp` — not a set of components to assemble.

It does **not** serve many users at once — a second conversation waits its turn in the queue. It does **not** survive a restart without a fresh prefill, and it does **not** switch between conversations without one. Those are the things session storage would add, and all are knowingly left for later.

---

## 6. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Session storage / persistence (`O_DIRECT` store, identity, eviction) | a separate capability with its own design; the near-term workload is one live conversation, which never leaves VRAM | prefix reuse and the server are green and concurrent conversations or restart recovery are actually needed |
| Prefix **matcher** (block table, cache key, radix search, eviction) | its parameters are measurements of an assembled graph; a single live session needs no key and nothing to match | session storage exists and a fork / shared-system-prompt workload exists |
| Multiple concurrent resident sessions | continuous batching territory; the engine is single-sequence by design | the queue proves to be the actual bottleneck |
