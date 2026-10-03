# Server reference analysis — what to adopt for G5

Date `2026-10-03`. Question: the server (G5) is the one product surface with almost nothing to share with the engine's own strategy, so can it be inherited from a reference rather than designed from scratch? Method: read of the two native servers among the references — llama.cpp's C++ server and ds4's C server — for their layering and their transport. No Aeon code changed.

**Two references, one layer.** Both are transport references: llama.cpp gives the C++ shape over `cpp-httplib`, ds4 the same thing hand-rolled over raw sockets. Neither is forked — the shell is a blueprint, and the engine adapter behind it is ours.

**Reference checkouts:** `9cf3bf2` at `/home/marcolap/aeon-references/llama.cpp` (shallow, default branch); `/home/marcolap/aeon-references/ds4` (default branch). Re-pull before trusting a specific revision.

**Verdict.** The **transport shell is worth inheriting as a blueprint and reused as a pattern**, not forked. The **engine adapter is ours by definition** — llama.cpp's is built on continuous batching and multi-slot parallelism, both of which our requirements exclude. This document records only the pieces we *should* reference, and stops where the reusable layer ends.

---

## 1. The layering we should mirror

llama.cpp's server already separates transport from engine, and the split is measurable by counting engine references per file. The shape it converged on is the same one G5 needs — which is independent evidence that the boundary is in the right place.

```mermaid
flowchart TD
    HTTP["HTTP / SSE transport<br/>server-http · cpp-httplib"]
    STREAM["Resumable SSE by conversation id<br/>server-stream"]
    QUEUE["Task queue + response reader<br/>server-queue"]
    ADAPTER["Engine adapter / session seam<br/>server-context (ours: the neutral seam)"]
    HTTP --> STREAM --> QUEUE --> ADAPTER
```

The reusable layer is everything **above** the adapter. The measurement:

| File | Engine refs | Layer |
| :--- | ---: | :--- |
| `server-queue.cpp/.h` | **0** | queue / result plumbing |
| `server-stream.cpp/.h` | **0** | streaming buffer |
| `server-http.cpp/.h` | 5 (UI assets only) | transport |
| `server-context.cpp/.h` | 147 | **adapter — not reusable** |

Two facts worth carrying into the plan: the queue and stream layers are *already engine-agnostic* (zero references), and the adapter is where every engine reference concentrates. That is exactly the G5/engine seam AGENTS.md now defines, and it tells us the shell can be studied in isolation.

---

## 2. Reference map — what we should reference

Each row names the file, the symbol to read, and what Aeon takes from it. These are the only llama.cpp server files worth citing for this work.

| Concern | File | Symbol | What we take |
| :--- | :--- | :--- | :--- |
| **Streaming response model** | [server-http.h](../../../../aeon-references/llama.cpp/tools/server/server-http.h) | `server_http_res` (`data`, `next`), `is_stream()`, `on_complete()` | A response is either a full body or a `next()` callback yielding chunks. This is the exact streaming seam G5 needs (R2), and it names no engine type. |
| **Transport abstraction** | [server-http.h](../../../../aeon-references/llama.cpp/tools/server/server-http.h) | `server_http_context` (`get`/`post`/`del`, `init`/`start`/`stop`), `handler_t` | Route registration and a server thread, over `cpp-httplib`. The shape to mirror for a single-endpoint server. |
| **Resumable SSE** | [server-stream.h](../../../../aeon-references/llama.cpp/tools/server/server-stream.h) | `stream_session`, `stream_pipe_producer::write`, `server_res_spipe` (tee) | A per-generation ring buffer that survives client disconnect, **keyed by conversation id** — "one conv = one live session", which matches our single-session model (R3/R4) almost exactly. |
| **Conversation-id convention** | [server-stream.h](../../../../aeon-references/llama.cpp/tools/server/server-stream.h) | `server_stream_conv_id_from_headers` (`X-Conversation-Id`) | A neutral, non-engine way to identify the live session across streaming reconnects. |
| **Task queue** | [server-queue.h](../../../../aeon-references/llama.cpp/tools/server/server-queue.h) | `server_queue` (`post`, `defer`, `start_loop`, `on_new_task`, `on_update_slots`) | A producer/consumer queue with deferred tasks — the template for R3's "the second conversation queues". |
| **Response reader + cancellation** | [server-queue.h](../../../../aeon-references/llama.cpp/tools/server/server-queue.h) | `server_response`, `server_response_reader` (`next`, `wait_for_all`, `stop`) | A generator-like consumer with a `should_stop` predicate per poll — the pattern for R4 cancellation. |
| **Adapter seam** | [server-context.h](../../../../aeon-references/llama.cpp/tools/server/server-context.h) | `server_context` (`load_model`, `start_loop`, `terminate`, `get_response_reader`, `get_meta`) | The lifecycle a long-lived engine needs: load once, block in a loop, terminate cleanly. The *interface shape* to copy; the *implementation* is ours. |
| **Launch shape** | [main.cpp](../../../../aeon-references/llama.cpp/tools/server/main.cpp) | `main` (5 lines) | Confirms the executable should be a trivial wrapper: parse args, hand to the app, run. Matches the thin-`main` discipline `aeon_chat` already follows. |
| **Target structure** | [CMakeLists.txt](../../../../aeon-references/llama.cpp/tools/server/CMakeLists.txt) | `server-context` (static lib), `llama-server-impl`, `llama-server` (exe) | The server is a separate target linking the engine, never the reverse — the build-level form of G5's one-way dependency (R7/R8). |
| **HTTP dependency** | [vendor/cpp-httplib](../../../../aeon-references/llama.cpp/vendor/cpp-httplib) | `httplib.h`, `httplib.cpp` | The pure C/C++ HTTP choice that satisfies R7: no runtime dependency, vendorable, no second stack. |
| **OpenAI-compatible surface** | [server-context.h](../../../../aeon-references/llama.cpp/tools/server/server-context.h) | the `server_routes` handler list | The endpoint *names and shapes* to match (`/v1/chat/completions`, `/health`, `/props`) so existing clients work unchanged — AGENTS.md rule 2, standard ecosystem compatibility. |

Two of these deserve emphasis. First, **`server_http_res`'s generator pattern** is the single most reusable idea: it decouples "how a response is produced" from "how it is transported", which is precisely the seam that lets the server stay model-agnostic. Second, **`server-stream`'s conversation-id-keyed buffer** is the closest existing analogue of our design intent, and it is already engine-free — worth reading in full before writing ours.

---

## 3. Where the reusable layer ends

The engine adapter (`server-context.cpp`, ~5,558 lines) is **not** a reference for us, and the reason is structural, not stylistic: it is built to run **many parallel sequences through one batched decode**. It carries `n_parallel` slots, per-slot `seq_id`s, a shared `llama_batch`, and multi-user scheduling. Our requirements exclude continuous batching and multi-session residency (out of scope), so almost all of that file describes a machine we deliberately do not have.

The same boundary excludes llama.cpp's `common/` support library, which the "generic-looking" server files actually depend on (tokenization, sampling chains, the Jinja chat-template engine, grammar). Inheriting those files would mean inheriting llama.cpp's own runtime — the opposite of a self-contained G5.

**Our adapter is small precisely because our scope is small.** Where llama.cpp spends thousands of lines on slot scheduling and feature breadth (embeddings, rerank, infill, multimodal, LoRA, router mode, embedded UI), G5 needs one conversation, one queue, one streaming endpoint. The adapter is ours to write, and it is the part that could never be inherited anyway.

Rule of thumb for the plan: **read the shell for patterns, write the adapter.** Do not fork the directory.

---

## 4. Second reference — ds4, a native C server

`ds4` is the other reference with a **native compiled server**: `ds4_server.c`, ~20.3k lines of C, raw `sys/socket` + `pthreads` with **no HTTP library at all**. Where llama.cpp shows the C++ shape over `cpp-httplib`, ds4 shows the same thing done by hand — useful evidence that the transport really is small (R2), and an independent endpoint surface to match (R1).

Its own header states the design:

> each client connection is handled by a small blocking thread that parses one request, then queues a job to a **resident session worker**. A model coordinator batches decode-ready sessions and serializes bounded prefill quanta, keeping graph mutations out of client threads while preserving **per-session KV ownership**.

| Concern | File | Symbol | What we take |
| :--- | :--- | :--- | :--- |
| **SSE, hand-rolled** | [ds4_server.c](../../../../aeon-references/ds4/ds4_server.c) | `sse_headers`, `sse_chunk`, `sse_done`, `sse_chat_delta_n` | A complete SSE implementation over a raw socket — the reference for *how little* transport code actually is when a library is not wanted (R2). |
| **OpenAI / Anthropic surface** | [ds4_server.c](../../../../aeon-references/ds4/ds4_server.c) | routes `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/models`, `/v1/responses` | Confirms the endpoint set R1 should match, including Anthropic Messages. |

One thing to leave:

- **Leave the batch coordinator.** It batches decode-ready sessions (continuous batching), which we exclude. Read its transport, skip its scheduler.

---

## 5. Deferred reference — session storage

Session storage — snapshotting a sequence's state and restoring it later — is a capability we **deferred**, not current work. Recorded here so it is not lost, not so it is started.

| Deferred capability | File | Symbol |
| :--- | :--- | :--- |
| Per-session KV snapshot + restore | [server-task.h](../../../../aeon-references/llama.cpp/tools/server/server-task.h) | `server_prompt_cache` (`alloc`, `load`, `update`), `server_prompt::prompt_save` / `prompt_load` |
| Save/restore endpoints | [server-context.h](../../../../aeon-references/llama.cpp/tools/server/server-context.h) | `handle_slots_save`, `handle_slots_restore`, `handle_slots_erase` |
| Disk store with eviction reasons | [ds4_kvstore.h](../../../../aeon-references/ds4/ds4_kvstore.h) | `ds4_kvstore_open` / `store_live_prefix` / `try_load_text` / `evict`, `DS4_KVSTORE_REASON_{COLD,CONTINUED,EVICT,SHUTDOWN}` |

llama.cpp's prompt cache snapshots to host RAM; ds4's kvstore persists to disk, keys entries by a content hash, tracks hits, and records *why* an entry was evicted — closer to a cold-tier store whose home must be NVMe, not the Warm RAM the experts already over-subscribe.

---

## 6. How this maps to the requirements

| Requirement | Backed by |
| :--- | :--- |
| R1 conversational endpoint, R2 streaming | `server_http_res` generator pattern; `server_routes` and ds4 route list; ds4 `sse_*` |
| R3 second conversation queues | `server_queue` (`post`/`defer`) |
| R4 cancellation | `server_response_reader::next` + `should_stop`; resumable stream |
| R5 robustness | `server_http_context` handler/response separation |
| R6 operability | ds4 `DS4_KVSTORE_REASON_CONTINUED` vs `COLD` — a ready vocabulary for reporting reuse |
| R7 one native binary, one command | `cpp-httplib` + the `llama-server` target/exe structure |
| R8 model-agnostic server | the measured llama.cpp seam: queue/stream/http hold zero engine references; only the adapter does |

---

## 7. One-line summary

Two transport references. **llama.cpp** gives the **C++ shell** — streaming-response generator, conversation-id-keyed resumable SSE, queue with cancellation, `cpp-httplib`, an OpenAI-shaped surface, and a target structure that encodes the one-way dependency — and confirms by measurement that the engine adapter is a separate layer. **ds4** gives the same layer hand-rolled — raw-socket SSE and its own route set — evidence that the transport is small and an independent surface to match. We adopt the patterns and the HTTP dependency; we write the **session seam** behind them ourselves, because ours is single-session where both references' schedulers are batching.
