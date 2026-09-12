# Native Text-In/Text-Out Implementation Plan

*Status: native text-in/text-out turn verified; external behavioral correctness gate remains open*

## 1. Objective

Complete Aeon as a usable inference engine for the local DeepSeek-V4-Flash-0731 checkpoint:

```text
human text/messages
    -> native DSV4 prompt formatting
    -> native tokenizer
    -> Aeon token-ID generation
    -> EOS-aware stopping
    -> native detokenizer
    -> human-readable response
```

The final runtime must remain C++20/native HIP with no Python, PyTorch, Transformers, or vLLM dependency. Python/vLLM may be used during preparation and validation as an offline behavioral oracle. The available machine cannot run the complete DSV4 checkpoint through vLLM locally, so final model-behavior comparison must use a compatible external API.

This work is a prerequisite for trustworthy activation profiling. The current `profile_routing` results are plumbing artifacts only and must not drive Hot/Warm placement until the prompt contract and native model behavior have passed the correctness gates below.

## 1.1 Current Implementation Status

- [x] Native versioned tokenizer artifact preparation and C++ ByteLevel-BPE loader.
- [x] Native DSV4 chat/thinking formatter with exact control-token and multi-turn parity tests.
- [x] EOS-aware greedy generation with explicit `eos`, `max_new_tokens`, `context_limit`, and `error` stop reasons.
- [x] `aeon_chat` native CLI and full 43-layer silicon smoke (`What is 2+2?`, 4 generated tokens, `5.57 tok/s`).
- [x] Full 43-layer unrestricted chat turn reached EOS with a readable answer (`What is the capital of France?` -> `The capital of France is **Paris**.`).
- [ ] External compatible-reference comparison and activation-profile unlock.

## 2. Current boundaries and evidence

### 2.1 Existing Aeon runtime

The core model execution path already accepts token IDs and returns token IDs:

- `src/architecture/deepseek_v4/core/v4_pipeline.hpp`
  - `V4Pipeline::step(uint32_t token_id, uint32_t pos, RoutingPhase phase)` performs one token step.
  - `V4Pipeline::generate(const std::vector<uint32_t>& prompt, uint32_t max_new_tokens, ...)` performs prompt prefill and autoregressive decoding.
  - The prompt loop is currently marked `RoutingPhase::Prefill`.
  - The decode loop is currently marked `RoutingPhase::Decode`.
  - The legacy `V4Pipeline::generate(...)` raw-ID API remains fixed-length and does not stop on EOS by itself.
  - The native text wrapper in `src/infrastructure/text/text_generation.*`, used by `aeon_chat`, adds EOS-aware stopping, context limits, and explicit stop reasons around token generation.
  - The first returned generated token is produced by the final prompt-prefill step.
- `src/infrastructure/core/routing_counter.hpp` and `src/infrastructure/core/routing_profile.hpp`
  - provide optional routing observation and durable profile aggregation;
  - must remain usable with token IDs after the text frontend is added.
- `tools/profile_routing.cpp`
  - is the current tokenized-corpus profiling driver;
  - should eventually accept tokenized prompts emitted by the native or offline text preparation path, but profiling remains blocked until correctness is established.
- `tests/test_hot_warm_cold_pipeline.cpp` and `tests/test_dynamic_expert_pool.cpp`
  - are existing native execution and full-model regression surfaces.

### 2.2 Model and tokenizer facts

The native model artifact is:

```text
models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/
```

Important local files:

- `config.json`
  - `model_type: deepseek_v4`;
  - `num_hidden_layers: 43`;
  - `n_routed_experts: 256`;
  - `num_experts_per_tok: 6`;
  - `vocab_size: 129280`;
  - `bos_token_id: 0`;
  - `eos_token_id: 1`.
- `tokenizer.json`
  - standard Hugging Face ByteLevel-BPE vocabulary, merges, added tokens, pre-tokenizer, and decoder data;
  - includes DSV4 control tokens.
- `tokenizer_config.json`
  - `add_bos_token: false`;
  - `add_eos_token: false`;
  - BOS token `<｜begin▁of▁sentence｜>` with ID `0`;
  - EOS token `<｜end▁of▁sentence｜>` with ID `1`;
  - no `chat_template` field;
  - `tokenizer_class: PreTrainedTokenizerFast`.
- `generation_config.json`
  - must be inspected and recorded as part of the generation contract when implementing sampling/defaults.

The absence of `chat_template` is not evidence that the checkpoint is completion-only. The local source snapshot identifies the model as `yiminyuan/DeepSeek-V4-Flash-0731`, documents chat usage, and includes a custom formatter under:

```text
models/DeepSeek-V4-Flash-0731-INT4-W4A16/
  models--yiminyuan--DeepSeek-V4-Flash-0731/
    snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0/
      encoding/encoding_dsv4.py
      encoding/test_encoding_dsv4.py
      encoding/README.md
      README.md
```

That formatter is the checkpoint-specific contract to reproduce. Do not assume ChatML, DeepSeek V2, or DeepSeek V3 formatting.

### 2.3 Reference implementations already identified

Use these paths as behavioral references; do not add them as runtime dependencies:

- vLLM source:
  - `/home/marcolap/src/vllm-023/vllm/tokenizers/deepseek_v4.py`;
  - `/home/marcolap/src/vllm-023/vllm/tokenizers/deepseek_v4_encoding.py`;
  - `/home/marcolap/src/vllm-023/vllm/renderers/deepseek_v4.py`;
  - `/home/marcolap/src/vllm-023/vllm/tokenizers/registry.py`;
  - `/home/marcolap/src/vllm-023/vllm/renderers/registry.py`.
- Installed preparation/reference environment:
  - `/home/marcolap/.venvs/vllm-023-rocm`;
  - `tokenizers 0.22.2`, `transformers 5.12.1`, `vllm 0.23.0`.
- llama.cpp checkout:
  - `/home/marcolap/aeon-references/llama.cpp`;
  - DSV4 conversion template: `conversion/deepseek.py` and `models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja`;
  - tokenizer API and implementation: `include/llama.h`, `src/llama-vocab.cpp`;
  - chat-template discovery/application: `src/llama-model.cpp`, `src/llama-chat.cpp`, and `common/chat.cpp`.

llama.cpp is the independent prompt-format cross-check. Its older fixed-template registry does not provide a dedicated DSV4 legacy entry; the relevant DSV4 path is its Jinja template embedded during GGUF conversion. The current Aeon `.aeon` format does not contain GGUF chat metadata, so the DSV4 formatter must be carried explicitly into Aeon.

## 3. Frozen DSV4 prompt contract

Implement the minimum chat contract first. Keep tool calls and advanced roles outside the first milestone until basic user/assistant text is correct.

Known control IDs from the local tokenizer:

```text
BOS                 <｜begin▁of▁sentence｜>   0
EOS                 <｜end▁of▁sentence｜>     1
USER                <｜User｜>                128803
ASSISTANT            <｜Assistant｜>           128804
THINK_OPEN          <think>                 128821
THINK_CLOSE         </think>                128822
```

The exact IDs must be read from the tokenizer artifact during preparation and asserted in fixtures; do not hardcode them without verification.

The first supported modes are:

### 3.1 Chat mode

The formatter must:

1. emit BOS exactly once at conversation start;
2. emit the user marker before each user message;
3. emit the assistant marker for the generation turn;
4. emit the DSV4 chat-mode thinking boundary expected by the bundled formatter, currently represented by `</think>` after the assistant marker;
5. leave the model to generate the assistant response;
6. stop on EOS ID `1`.

For the known simple case, the expected serialized prompt is conceptually:

```text
<BOS><｜User｜>Hello<｜Assistant｜></think>
```

with expected token IDs:

```text
[0, 128803, 19923, 128804, 128822]
```

This fixture must be reproduced exactly by the native formatter/tokenizer.

### 3.2 Thinking mode

The formatter must support the explicit thinking variant separately:

```text
<BOS><｜User｜>Hello<｜Assistant｜><think>
```

The model may emit reasoning content, close it with `</think>`, and then emit the answer. Do not merge chat and thinking mode in one implicit behavior. Store the selected mode in every text-generation and profiling artifact.

### 3.3 EOS and stop behavior

The first generated token is obtained during the final prompt step. Therefore EOS must be checked:

- immediately after the final prefill step;
- after every decode step;
- before feeding an EOS token back through the model.

The native API must report a stop reason:

```text
eos
max_new_tokens
context_limit
error
```

The first implementation may expose greedy decoding only. Sampling, tool-call stops, and streaming token callbacks are later milestones.

## 4. Native module design

Keep responsibilities separate. Do not grow `v4_pipeline.hpp` with tokenizer and formatting logic.

Recommended modules:

```text
src/architecture/deepseek_v4/text/
    dsv4_tokenizer.hpp / dsv4_tokenizer.cpp
    dsv4_chat_formatter.hpp / dsv4_chat_formatter.cpp
src/infrastructure/text/
    text_generation.hpp / text_generation.cpp
```

If the project continues its header-heavy style for the first slice, use focused header-only modules temporarily, but preserve these ownership boundaries.

### 4.1 `Dsv4Tokenizer`

Responsibilities:

- load a native tokenizer artifact derived from standard `tokenizer.json`;
- encode UTF-8 text to token IDs;
- decode token IDs back to UTF-8 text;
- expose vocabulary size and verified special-token IDs;
- distinguish literal control tokens from ordinary text;
- support `add_special_tokens=false` internally so the formatter controls BOS explicitly;
- preserve ByteLevel behavior, BPE merges, added tokens, pre-tokenizer rules, and decoder behavior.

Do not implement a reduced word-split tokenizer. It would produce plausible-looking IDs but invalidate the model and routing measurements.

### 4.2 Tokenizer artifact strategy

Use a preparation-time converter rather than parsing arbitrary JSON in the inference hot path:

1. Read the standard model `tokenizer.json` using the installed Python `tokenizers` package.
2. Validate the tokenizer against golden strings and the known DSV4 control IDs.
3. Emit a versioned native artifact, for example:

```text
models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon
```

4. Load `tokenizer.aeon` from C++ at runtime.
5. Record the source `tokenizer.json` hash, tokenizer artifact version, and formatter version in metadata.

The artifact must contain at least:

- format magic/version;
- vocabulary entries and token flags;
- added-token table;
- merge ranks;
- pre-tokenizer/decoder configuration needed by this model;
- special-token IDs;
- source tokenizer hash.

The converter may initially be a Python script under `scripts/`, because it is a model-preparation tool. The runtime must not import Python.

Alternative implementation, only if artifact generation becomes a blocker: port the required ByteLevel-BPE loader directly to C++ and load `tokenizer.json` using a small structured parser. This is less desirable because it expands runtime parsing surface and complicates validation.

### 4.3 `Dsv4ChatFormatter`

Responsibilities:

- accept a typed message list, initially `{role: system|user|assistant, content}`;
- render chat mode and thinking mode according to the DSV4 formatter;
- insert BOS and role markers exactly once and in the correct order;
- return both rendered control/text representation and token IDs for diagnostics;
- reject unsupported roles and malformed message history explicitly;
- keep tool-call formatting out of the first milestone but reserve an extension point.

The formatter should be deterministic and independent of GPU code. Golden tests must compare its output against the bundled `encoding_dsv4.py` and vLLM formatter for identical message lists.

### 4.4 EOS-aware pipeline API

Do not replace the existing token-ID API. Add a new layer around it:

```cpp
struct GenerationOptions {
    uint32_t max_new_tokens{256};
    uint32_t eos_token_id{1};
    bool stop_on_eos{true};
    bool thinking_mode{false};
};

struct GenerationResult {
    std::vector<uint32_t> token_ids;
    StopReason stop_reason;
};

GenerationResult generate_until_stop(
    const std::vector<uint32_t>& prompt,
    const GenerationOptions& options
);
```

The existing `generate()` behavior should remain available for regression tests. Implement `generate_until_stop()` either as a new method or by adding an explicit options path, then make the text API use the EOS-aware path.

Required safeguards:

- reject an empty prompt;
- reject a zero generation limit;
- ensure prompt plus generation limit fits context capacity;
- check EOS from the final prefill result;
- stop before another `step()` after EOS;
- preserve generated token IDs for diagnostics;
- preserve existing greedy argmax behavior for the first correctness milestone.

### 4.5 Native text API and executable

Add a small executable, tentatively:

```text
tools/aeon_chat.cpp
```

Initial responsibilities:

- load native model and tokenizer artifact;
- accept a plain prompt or a small JSONL message record;
- select chat/thinking mode explicitly;
- format and tokenize the messages;
- invoke EOS-aware generation;
- detokenize the generated IDs;
- print rendered prompt, prompt IDs, generated IDs, stop reason, and response text when diagnostic mode is enabled;
- return a nonzero code for tokenizer, formatting, model, or context errors.

Keep an API-independent service layer behind the executable. A later HTTP/OpenAI-compatible server should call the same text service rather than duplicate formatting or generation logic.

## 5. Preparation and reference fixtures

Before implementing GPU-facing text serving, produce a small golden fixture set with the `vllm-023-rocm` environment and the bundled formatter. Store only compact, non-model artifacts in the repository, for example:

```text
tests/fixtures/dsv4_text/
    basic_chat.json
    thinking_chat.json
    tokenizer_golden.json
    expected_prompt_ids.json
```

Each fixture should include:

- message list;
- mode;
- rendered prompt/control-token representation;
- exact prompt IDs;
- expected detokenization round trips;
- BOS/EOS IDs;
- formatter/tokenizer source hashes.

Required fixture cases:

1. one user message: `Hello`;
2. a longer technical question;
3. an empty or whitespace-sensitive message;
4. multi-turn user/assistant/user history;
5. explicit thinking mode;
6. literal control-token text that must not be confused with formatter-inserted control tokens;
7. EOS and decode round-trip cases.

The fixtures are a contract, not a claim that the full model can run locally in vLLM.

## 6. Correctness validation without local vLLM inference

The machine cannot run the full DSV4 checkpoint through vLLM, so use layered validation.

### Gate A: tokenizer and formatter parity

Run the Python preparation/reference tools and native Aeon against the same fixtures. Require:

- exact prompt token IDs;
- exact special-token IDs;
- exact decoded text for controlled token sequences;
- exact BOS/EOS insertion behavior;
- explicit mode agreement.

The Python environment is only a reference/preparation dependency. It must not be linked into CMake or loaded by native serving.

### Gate B: local native component correctness

Before interpreting any answer, retain and expand focused tests for:

- tokenizer encode/decode;
- BPE merge behavior;
- ByteLevel whitespace and punctuation behavior;
- added-token recognition;
- DSV4 formatter state transitions;
- multi-turn formatting;
- EOS stop behavior;
- context-limit behavior;
- deterministic greedy generation loop.

Use the existing native model tests for transformer and swizzled-kernel correctness:

- `test_w4a16_swizzle`;
- `test_w4a16_swizzled_gemv`;
- `test_w4a16_swizzled_dual_gemv`;
- `test_aeon_moe_fused_w13`;
- `test_aeon_moe_fused_w2`;
- `test_v4_attention`;
- `test_v4_block`;
- `test_hot_warm_cold_pipeline`;
- `test_hot_warm_cold_pipeline`.

### Gate C: external API behavioral comparison

Use a compatible external API as the behavioral oracle. The API provider/model revision is part of every result artifact. Use deterministic settings where available:

```text
temperature = 0
top_p = 1
fixed maximum output tokens
explicit system/user messages
explicit DSV4 mode if the provider exposes it
```

Compare a fixed correctness corpus containing:

- simple factual questions;
- arithmetic and exact-format tasks;
- instruction following;
- code generation;
- multi-turn context;
- a longer answer requiring more than eight tokens;
- EOS/length behavior.

Record:

- input messages and selected mode;
- exact Aeon rendered prompt and token IDs;
- Aeon generated IDs, stop reason, and decoded text;
- API request parameters, model identifier, timestamp, and response;
- provider logprobs or token IDs when available;
- manual/evaluated behavioral result.

An API match is a behavioral check, not proof of exact logits or expert routing. Differences may arise from model revision, quantization, sampling implementation, hidden system prompts, or provider formatting. Do not claim activation-level parity from API text agreement alone.

### Gate D: activation-profile eligibility

Only after Gates A-C pass for the chosen prompt mode:

- mark the prompt contract as verified;
- generate the profile corpus with the same formatter and tokenizer;
- unlock multi-prompt routing profiling;
- keep profile and held-out corpora separate;
- record the tokenizer/formatter hashes in `profile_routing` metadata.

## 7. API and future serving architecture

The first native executable should be transport-independent. Later serving should layer on:

```text
HTTP/OpenAI-compatible request
    -> request/message validation
    -> DSV4 formatter
    -> tokenizer
    -> generation scheduler
    -> detokenizer/streamer
    -> response
```

The API layer must not own model-specific prompt serialization. It calls the same `Dsv4ChatFormatter` and `TextGenerationService` used by the CLI and tests.

Future serving requirements:

- streaming token output;
- request cancellation;
- bounded context and generation limits;
- one-request baseline before concurrency;
- explicit model/template version in health and metadata responses;
- deterministic request logging without leaking prompt contents by default;
- later multi-request scheduling and KV-cache policy.

Do not begin multi-request scheduling before single-request text correctness is established.

## 8. Implementation sequence and gates

### Step 0: freeze evidence and contract

- Record model directory hashes, tokenizer hash, local snapshot revision, and formatter source hash.
- Preserve the existing token-ID generation tests.
- Add the DSV4 fixture format and Python fixture generator.
- Decision gate: known prompt IDs and special-token behavior are reproducible.

### Step 1: native tokenizer artifact and C++ loader

- Implement the preparation converter from `tokenizer.json` to `tokenizer.aeon`.
- Implement native load, encode, decode, and special-token accessors.
- Add CPU-only tokenizer tests before connecting GPU inference.
- Decision gate: exact fixture parity with Python.

### Step 2: native DSV4 formatter

- Implement typed message validation and chat/thinking rendering.
- Add multi-turn and control-token fixtures.
- Keep tool-call roles unsupported but explicitly rejected.
- Decision gate: exact formatted prompt and token-ID parity.

### Step 3: EOS-aware token generation

- Add `GenerationOptions`, `GenerationResult`, and stop reasons.
- Check EOS after prefill and every decode step.
- Add a test that forces or supplies an EOS result and verifies no extra step occurs.
- Add a longer generation limit to the CLI; remove the misleading eight-token-only demonstration.
- Decision gate: generated ID sequence and stop reason are deterministic.

### Step 4: native text CLI

- Add `tools/aeon_chat.cpp` and CMake target.
- Print diagnostic prompt IDs, generated IDs, decoded text, and stop reason.
- Run one real user question through all 43 layers.
- Decision gate: human-readable response is produced without Python at runtime.

### Step 5: external API comparison

- Create a correctness corpus and a small API adapter outside the runtime.
- Compare Aeon and API under controlled settings.
- Investigate failures before changing profiler or cache behavior.
- Decision gate: agreed behavioral threshold and no unexplained prompt-format mismatch.

### Step 6: unlock activation profiling

- Update `profile_routing` to consume verified text-derived JSONL or expose a text-to-profile preparation command.
- Add tokenizer/formatter hashes and prompt mode to profile metadata.
- Build profile and held-out corpora.
- Resume the per-layer probability ranking study.

### Step 7: serving transport

- Add a transport layer around the tested text-generation service.
- Start with one request at a time and streaming output.
- Add concurrency only after single-request behavior and cancellation are stable.

## 9. Files to add or change

Expected focused implementation surface:

```text
scripts/prepare_dsv4_tokenizer.py
src/architecture/deepseek_v4/text/dsv4_tokenizer.hpp
src/architecture/deepseek_v4/text/dsv4_tokenizer.cpp
src/architecture/deepseek_v4/text/dsv4_chat_formatter.hpp
src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp
src/infrastructure/text/text_generation.hpp
src/infrastructure/text/text_generation.cpp
tools/aeon_chat.cpp
tests/test_dsv4_tokenizer.cpp
tests/test_dsv4_chat_formatter.cpp
tests/test_text_generation.cpp
tests/fixtures/dsv4_text/*
CMakeLists.txt
plans-and-docs/execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md
AGENTS.md
```

Do not modify `bench_full_model.cpp` for this feature. Do not make Python a CMake or runtime dependency. Do not introduce a generic chat-template fallback.

## 10. Definition of done

The text-in/text-out milestone is complete when:

1. A native executable accepts a normal human-language prompt.
2. The prompt is formatted according to the verified DSV4 contract.
3. Native tokenization produces fixture-identical IDs.
4. The full 43-layer Aeon pipeline generates token IDs without a fixed eight-token cap.
5. EOS and context-limit stopping are explicit and tested.
6. Native detokenization returns human-readable text.
7. The executable reports a response and stop reason without Python installed or imported at runtime.
8. Component and formatter tests pass.
9. The output is compared against an external compatible API under recorded deterministic settings.
10. Any remaining differences are documented and explained to the extent possible given API-only reference access.
11. Only then are multi-prompt activation rankings eligible for placement decisions.

## 11. Important limitations

- External API agreement cannot prove exact hidden-state, logits, or expert-routing parity.
- The API may use a different checkpoint revision, quantization, system prompt, or formatter.
- The local checkpoint's missing standard `chat_template` remains a packaging/provenance issue; the custom DSV4 formatter is the current evidence-backed contract.
- Native tokenizer correctness is a hard prerequisite. A simplified tokenizer invalidates both responses and activation profiles.
- Chat and thinking modes must remain separate datasets and metadata identities.
- Tool calling, sampling, batching, cancellation, and OpenAI compatibility are later layers, not excuses to postpone the core native text path.
