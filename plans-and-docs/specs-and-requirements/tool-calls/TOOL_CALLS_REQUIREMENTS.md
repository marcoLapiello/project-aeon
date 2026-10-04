# Full Tool-Call Support — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** the **outbound half of the tool-call loop** — reading the model's own tool-call block back into the structured OpenAI `tool_calls` an agent client waits on, both non-streaming and streamed. The inbound half (tool definitions sent to the model, and tool results rendered back in) already exists in the prompt encoder; this document covers the return path and the round trip, at the G4/G5 seam.

**See also:** [Server requirements](../server/SERVER_REQUIREMENTS.md) R3/R5 and its §6 deferred row that names this work. [Server reference analysis](../../analysis/current/SERVER_REFERENCE_ANALYSIS.md) — how the reference servers expose tools over the wire. The prompt encoder (`dsv4_prompt_encoder.cpp`) is the authority for the DSML grammar this parses.

---

## 1. What we are building

An **agent client** (pi, qwen, and any OpenAI-compatible agent) declares tools, sends a conversation, and — when the model decides to act — expects a structured **`tool_calls`** array back, with `finish_reason: "tool_calls"`. It then executes the tools itself and sends the results back as `tool` messages.

Today the engine can already *send* tools to the model and *accept* tool results, but the model's own tool call — a `<｜DSML｜tool_calls>` text block the model writes into its reply — is returned **verbatim as `content`**. Plain chat works; an agent that waits on a `tool_calls` field sees none and stalls.

This work closes that loop: the DSML block the model emits is recognised and returned as the structured `tool_calls` the client expects, in both the non-streaming and streamed responses, without leaking raw DSML into `content`.

The workload remains the one the server is built for: **one live conversation**, driven by a human or an agent.

---

## 2. Requirements

**R1 — The model's tool call is returned as structured `tool_calls`.**
When the model emits a tool-call block, the non-streaming response carries `message.tool_calls` as an array of `{id, type: "function", function: {name, arguments}}`, where `arguments` is a JSON string, and `finish_reason` is `"tool_calls"`. `content` is empty or null for that turn. The raw DSML markup is not delivered as text.

**R2 — Streaming carries tool calls as deltas.**
In a streamed response the tool call arrives as `delta.tool_calls` entries (indexed, with `id` and `function.name` on first sight and `function.arguments` accumulated across deltas), the stream ends with `finish_reason: "tool_calls"`, and — critically — **no fragment of the DSML markup is ever emitted on the `content` channel**. A client that only reads `content` must never see `<｜DSML｜…>` text.

**R3 — The round trip is exact.**
A tool call this server emits, re-sent by the client as an assistant `tool_calls` message, re-renders through the prompt encoder to the **same prompt** the encoder would have produced for the model's original reply. The parse is the inverse of the encoder's render; the two are checked against each other (anti-circularity), not each against a hand-written fixture alone.

**R4 — A malformed or truncated block never becomes a bad tool call.**
A block that does not parse — truncated by the token cap, a deviating parameter form, an unknown element — degrades to the existing behaviour (the text stays in `content`) rather than emitting a partial, empty, or garbage `tool_calls` entry, and never raises a request error. Detection is all-or-nothing: either a well-formed block is parsed, or nothing is.

**R5 — Multiple invocations and typed arguments.**
A single block may contain more than one `<invoke>`; each becomes its own `tool_calls` entry with a distinct generated id. Parameter values are reconstructed by type: `string="true"` values pass through literally, `string="false"` values are parsed as JSON (number, boolean, array, object), matching the encoder's `encode_arguments_to_dsml`.

**R6 — Standard OpenAI surface only.**
The behaviour is driven entirely by the standard request fields a client already sends — `tools`, `tool_choice`, the `tool` role and its `tool_call_id`, and `finish_reason`. No client-specific flag or vLLM-only field is required to get tool calls. Tool calls are returned whenever the model produces them; whether the model is nudged toward, away from, or forced into a call is `tool_choice`'s job, and out of scope here beyond accepting it.

**R7 — Prefix reuse survives a tool turn.**
A conversation that includes a tool call and its result still reuses the resident prefix on the next turn. Because reuse is keyed on the exact ids an assistant turn produced, the tool-call body must be part of that turn's identity; parsing a tool call out of `content` must not silently break the reuse key.

**R8 — Model- and kernel-agnostic placement.**
The work lives in the G4 stream decoder / conversation adapter and the G5 codec only. It names no engine, weight-format, or kernel type; the DSML grammar is model knowledge and stays in G4, the OpenAI shape is transport knowledge and stays in G5. The serving layer still links no HIP runtime.

---

## 3. In scope

- Parsing the model's `<｜DSML｜tool_calls>` block into OpenAI `tool_calls` (non-streaming) and into streamed `tool_calls` deltas, with `finish_reason: "tool_calls"`.
- The inverse of the encoder's DSML rendering, including multiple invokes and typed (`string="true|false"`) parameters, verified against the encoder on a round trip.
- Keeping raw DSML off the `content` channel in both response shapes.
- The all-or-nothing fallback to text for a block that does not parse.
- Preserving prefix reuse across a tool turn (the turn's identity includes its tool-call bodies).
- Accepting the standard tool surface: `tools`, `tool_choice`, `tool` role + `tool_call_id`.

---

## 4. Out of scope

- **Tool execution.** The server never runs a tool; it returns the call and waits for the client to send the result back. This is the client's job by the OpenAI contract.
- **Constrained/grammar decoding of the tool call.** Forcing the model's tokens to conform to the tool schema with a logit mask is a separate capability; here the model writes the block in its own words and we parse it.
- **`tool_choice` semantics** beyond accepting the field — steering or forcing a call, and the `required`/named-function behaviours, are a model-steering concern.
- **Parallel tool calls as a server-side policy** — emitting whatever the model produced is in scope (R5); deciding *how many* the model should produce is not.
- **Any change to the model, its graph, the expert tier, or the prompt encoder's rendering.** The encoder's output is fixed; this work reads it back.
- **Multi-modal tool results** (images, files) — text and JSON results only, as the encoder already renders.

---

## 5. What this buys, and what it honestly does not

It turns the server from a **chat** endpoint into an **agent** endpoint: an OpenAI-compatible agent that declares tools, waits for a `tool_calls` field, executes, and loops will work against Aeon without a bespoke adapter. Combined with prefix reuse, a multi-step agent loop pays only for each step's addition, which is where a long tool-driven session is actually won or lost.

It does **not** make the server execute tools, and it does **not** constrain the model to the tool schema — if the model writes a malformed block, the client gets text, not a call. Both are stated, not hidden.

---

## 6. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Constrained decoding of tool arguments (logit mask against the schema) | a separate capability built on the sampler's logit-processor seam; the base loop parses what the model writes | clients need guaranteed-schema arguments, not best-effort parsing |
| `tool_choice`-driven steering / forcing | model-steering, not the return path; the return path works with `auto` | an agent needs `required` or a forced function |
| Multi-modal tool results | the encoder renders text/JSON tool results | a tool must return an image or a file |
| A second transport exposing tools (e.g. a Responses-style API) | the OpenAI chat-completions shape covers the near-term clients | a client speaks only the other shape |
