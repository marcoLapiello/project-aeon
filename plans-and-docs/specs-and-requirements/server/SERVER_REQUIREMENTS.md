# Server — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** the **server** — the production interface a real user drives: transport, request queue and access policy, streaming, cancellation, and the external contract. This is the G5 serving concern, above the engine. It is deliberately independent of prefix reuse: the server works over a stateless engine (see the [prefix reuse requirements](../prefix-reuse/PREFIX_REUSE_REQUIREMENTS.md)).

**See also:** [Server reference analysis](../../analysis/current/SERVER_REFERENCE_ANALYSIS.md) — the reference server whose transport shell G5 draws on, and where the reusable layer ends.

**Requirement ids** (R7–R14) are preserved from the document this was split out of, so the reference analyses keep resolving.

---

## 1. What we are building

A **server** that lets a person or a program hold one conversation over a network — send a message, receive the reply as it is generated — instead of invoking a CLI once per prompt.

The workload is the same single, explicit one the engine is built for: **one live conversation**, whether it is a human chatting over many turns or an agent extending its own context with tool results.

---

## 2. Requirements

**R7 — A conversational endpoint.**
The server accepts a conversation (messages plus the non-token inputs of the engine) and returns the model's reply. This is the same conversation abstraction the engine already consumes — the server adds transport, not a second notion of "conversation."

**R8 — Streaming.**
Tokens reach the client as they are generated. The client sees the reply being produced, not a final blob after a multi-second wait. First-token latency matters as much as total latency.

**R9 — One conversation at a time, by design; a second one queues.**
Exactly one live session is resident. A second conversation that arrives while one is live is **queued**, not refused. The queue is a fairness and turn-taking policy: a queued conversation is a *pending fresh conversation*, not a saved session — since only one session is resident and there is no session storage, when its turn arrives it prefills from scratch. Nothing is silently interleaved, and a queued request never corrupts the live session. The queue is the seam a future multi-session scheduler would grow from; single-session is a deliberate design choice, not a limitation to be worked around in this campaign.

**R10 — Cancellation.**
A generation in flight can be stopped by the client. Cancellation leaves the server and the engine in a well-defined state, and does not corrupt the live session.

**R11 — Robustness.**
Malformed input, a request too long for the context, an interrupted connection, an internal failure — each produces a defined error response and leaves the server running and the engine coherent. The server never crashes on bad input and never answers with a corrupt result.

**R12 — Operability.**
The operator can see: whether a session is live, what it currently holds, whether the last request reused resident state or recomputed (when the engine reports that decision), and the last request's timings. Enough to answer "why was that slow" without attaching a debugger.

**R13 — The product installs as one piece.**
The end user obtains Aeon as a **single product** and starts serving with **one command** — the way `vllm serve` or `llama-server` work. The server is not a separately installed component with its own dependency set: it builds and ships with the engine. The engine's zero-runtime-dependency discipline (AGENTS.md rule 1) is a property of **the whole product the user installs**, not of the engine binary alone — a user must never have to assemble the pieces or satisfy a second dependency stack to get a working server.

**R14 — The server is model-agnostic.**
The server holds **no model, weight-format, or kernel knowledge** (G5 depends on no G3/G4 type). The engine presents a conversation and streams back **decoded text** plus neutral scalars (stop reason, timings, token counts); the server never tokenizes, detokenizes, or renders a chat template. This is what lets a future model architecture or a second transport be added without touching the other side, and it fixes the dependency direction: the engine never depends on the server. The same neutral seam is what the tests and the CLI use, so "one conversation" has exactly one definition in the codebase.

---

## 3. In scope

- A server that serializes access to the single session: request intake, queuing, streaming output, cancellation, defined errors, basic operability (R7–R12).
- The **neutral conversation seam** consumed from the engine — conversation in, decoded text streamed out, with neutral scalars — so the server carries no model knowledge and the engine never depends on the server (R14).
- A **single build and install** that produces the engine and the server as one product, launched with one command (R13).
- The external contract: the endpoint shapes and the OpenAI-compatible surface, so existing clients work unchanged.

---

## 4. Out of scope

- **Prefix reuse itself** — the engine-side capability. It is a separate concern with its own document ([prefix reuse requirements](../prefix-reuse/PREFIX_REUSE_REQUIREMENTS.md)); the server does not require it to exist.
- **Continuous batching** — multiple sequences stepping together through the graph. Explicitly excluded; the engine is single-sequence by design.
- **Multiple concurrent sessions** — more than one live conversation resident at once. A second conversation queues (R9); it does not run concurrently.
- **Session storage / persistence** — writing session state to SSD and reloading it. On restart, or on switching conversations, the answer is a fresh prefill.
- **Prefix matching across sessions** — radix trees, block tables, cache keys, eviction, shared or forked prefixes.
- **Any change to the model, its graph, or the expert tier's strategy.**
- **Speculative decoding / MTP**, multi-GPU parallelism, KV-precision changes, tool *execution* beyond prompt encoding.

---

## 5. What this buys, and what it honestly does not

It gives the working engine a **real product shape**: one install, one command to serve, the same way a user expects of `vllm` or `llama.cpp` — not a set of components to assemble. A human or a program can hold one conversation over a network and watch the reply appear as it is generated.

It does **not** serve many users at once — a second conversation waits its turn in the queue. It does **not** make a turn cheap: without the engine's prefix reuse, every turn prefills from scratch, and the server reports that cost honestly rather than hiding it.

---

## 6. Relation to prefix reuse — independent

The server and prefix reuse are **two separate concerns**, each usable without the other:

- **The server can be built with no prefix reuse.** Over a stateless engine, every turn prefills from scratch and the server still works correctly — it is merely slower per turn.
- **Prefix reuse can be built with no server.** It is validated at the engine level, driven by the tests and the CLI.

They meet only at the **neutral conversation seam**: the server drives whatever the engine exposes, whether or not that engine reuses resident state. This is why the two have separate requirement documents and separate execution plans.

---

## 7. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Multiple concurrent resident sessions | continuous batching territory; the engine is single-sequence by design | the queue proves to be the actual bottleneck |
| Session storage / persistence | a separate capability; the server does not require it | concurrent conversations or restart recovery are actually needed |
| A second transport (e.g. gRPC) or auth | not needed for the single-user product shape | a real deployment needs it |
