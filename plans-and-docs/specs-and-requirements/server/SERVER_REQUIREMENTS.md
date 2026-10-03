# Server — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** the **server** — the production interface a real user drives: transport, request queue and access policy, streaming, cancellation, and the external contract. This is the G5 serving concern, above the engine.

**See also:** [Server reference analysis](../../analysis/current/SERVER_REFERENCE_ANALYSIS.md) — the reference server whose transport shell G5 draws on, and where the reusable layer ends.

---

## 1. What we are building

A **server** that lets a person or a program hold one conversation over a network — send a message, receive the reply as it is generated — instead of invoking a CLI once per prompt.

The workload is the same single, explicit one the engine is built for: **one live conversation**, whether it is a human chatting over many turns or an agent extending its own context with tool results.

---

## 2. Requirements

**R1 — A conversational endpoint.**
The server accepts a conversation (messages plus the non-token inputs of the engine) and returns the model's reply. This is the same conversation abstraction the engine already consumes — the server adds transport, not a second notion of "conversation."

**R2 — Streaming.**
Tokens reach the client as they are generated. The client sees the reply being produced, not a final blob after a multi-second wait. First-token latency matters as much as total latency.

**R3 — One conversation at a time, by design; a second one queues.**
Exactly one live conversation is served at a time. A second conversation that arrives while one is live is **queued**, not refused. The queue is a fairness and turn-taking policy; when a queued conversation's turn arrives, it is served. Nothing is silently interleaved, and a queued request never corrupts the live session. The queue is the seam a future multi-session scheduler would grow from; single-session is a deliberate design choice, not a limitation to be worked around in this campaign.

**R4 — Cancellation.**
A generation in flight can be stopped by the client. Cancellation leaves the server and the engine in a well-defined state, and does not corrupt the live session.

**R5 — Robustness.**
Malformed input, a request too long for the context, an interrupted connection, an internal failure — each produces a defined error response and leaves the server running and the engine coherent. The server never crashes on bad input and never answers with a corrupt result.

**R6 — Operability.**
The operator can see: whether a conversation is live, what it currently holds, and the last request's timings. Enough to answer "why was that slow" without attaching a debugger.

**R7 — The product installs as one piece.**
The end user obtains Aeon as a **single product** and starts serving with **one command** — the way `vllm serve` or `llama-server` work. The server is not a separately installed component with its own dependency set: it builds and ships with the engine. The engine's zero-runtime-dependency discipline (AGENTS.md rule 1) is a property of **the whole product the user installs**, not of the engine binary alone — a user must never have to assemble the pieces or satisfy a second dependency stack to get a working server.

**R8 — The server is model-agnostic.**
The server holds **no model, weight-format, or kernel knowledge** (G5 depends on no G3/G4 type). The engine presents a conversation and streams back **decoded text** plus neutral scalars (stop reason, timings, token counts); the server never tokenizes, detokenizes, or renders a chat template. This is what lets a future model architecture or a second transport be added without touching the other side, and it fixes the dependency direction: the engine never depends on the server. The same neutral seam is what the tests and the CLI use, so "one conversation" has exactly one definition in the codebase.

---

## 3. In scope

- A server that serializes access to the single session: request intake, queuing, streaming output, cancellation, defined errors, basic operability (R1–R6).
- The **neutral conversation seam** consumed from the engine — conversation in, decoded text streamed out, with neutral scalars — so the server carries no model knowledge and the engine never depends on the server (R8).
- A **single build and install** that produces the engine and the server as one product, launched with one command (R7).
- The external contract: the endpoint shapes and the OpenAI-compatible surface, so existing clients work unchanged.

---

## 4. Out of scope

- **Continuous batching** — multiple sequences stepping together through the graph. Explicitly excluded; the engine is single-sequence by design.
- **Multiple concurrent conversations** — more than one live conversation resident at once. A second conversation queues (R3); it does not run concurrently.
- **Multi-tenant access control** (authentication, per-user quotas) — the product shape for this revision is a single-user local server.
- **Any change to the model, its graph, or the expert tier's strategy.**
- **Speculative decoding / MTP**, multi-GPU parallelism, KV-precision changes, tool *execution* beyond prompt encoding.

---

## 5. What this buys, and what it honestly does not

It gives the working engine a **real product shape**: one install, one command to serve, the same way a user expects of `vllm` or `llama.cpp` — not a set of components to assemble. A human or a program can hold one conversation over a network and watch the reply appear as it is generated.

It does **not** serve many users at once — a second conversation waits its turn in the queue.

---

## 6. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Multiple concurrent resident sessions | continuous batching territory; the engine is single-sequence by design | the queue proves to be the actual bottleneck |
| Session storage / persistence | a separate capability; the server does not require it | concurrent conversations or restart recovery are actually needed |
| A second transport (e.g. gRPC) or auth | not needed for the single-user product shape | a real deployment needs it |
