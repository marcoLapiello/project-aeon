#!/usr/bin/env python3
"""Mutation sweep for the text binding (composition plan P4).

Per the plan's second rule: for each property the gate certifies, inject the
specific wrong variant into the *implementation* and require the gate to go red.
A survivor is either a gate defect to repair or an equivalent mutation to name as
such — never something to ignore.

Every mutation here targets `core/v4_engine.hpp`, because the binding is the whole
of what P4 adds: the tokenizer, the encoder, the generation loop, the graph and the
sampler are each owned and gated by their own phase, and mutating one of them would
prove something about *that* gate rather than about this one. What is under test is
whether `chat` composes them the way it claims — resets, reseeds, positions,
bounds, strips — and whether the artifact's policy is actually consulted.

Each mutation is expected to be killed by a *named* section, and the sweep prints
which checks failed so the section can be read off the output rather than trusted.

**One named equivalent is included on purpose**, because discovering it is worth
recording and it says something good about the layer state rather than something
bad about the gate. `EN-1` removes `chat`'s `reset_generation_state()` call. That
call is *redundant given the current state design*: `run_layer_body_decoding`
calls `layer.record_position(pos)` **before any read** (v4_layer_body.hpp:595), and
record_position re-derives every count from the position — `local_valid_count_`,
`compressor_partial_count_`, `compressed_entry_count_`, `indexer_candidate_count_`.
Every ring read is gated by one of those counts, so a fresh sequence starting at
position 0 overwrites exactly the slots it later reads and never touches a stale
row. That is requirement **R2** ("addressed by absolute position") doing real work,
not an accident.

The reset is **kept anyway**, and that is the point: it is the sequence-start
contract, and relying on every consumer being count-gated would put the safety
property in each consumer instead of at the boundary. Naming it an equivalent is
the honest reading — a survivor that is neither repaired nor named is the only
thing that would be wrong here.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/architecture/deepseek_v4/core/v4_engine.hpp"
BACKUP = Path("/tmp/aeon_engine_original.hpp")
TARGET = "test_v4_engine"

MUTATIONS = [
    (
        "EN-1 chat does not reset the generation state (named equivalent)",
        """        host_.reset_generation_state();
        // `set_config` validates and seeds the generator, so installing the config
        // is also the reseed — two calls would be two places for the seed to be
        // set and one place for them to disagree.
        sampler_->set_config(sampling);""",
        """        sampler_->set_config(sampling);""",
        False,   # see the module docstring: position-addressed state makes it a no-op
    ),
    (
        "EN-2 chat does not install the caller's sampling config",
        """        sampler_->set_config(sampling);

        text::GenerationOptions loop_options = generation;""",
        """        text::GenerationOptions loop_options = generation;""",
        True,
    ),
    (
        "EN-3 advance feeds every token at position 0",
        """        const half* logits = graph_->forward_token(token_id, position, host_.streams().compute);
        return sampler_->select(logits, host_.streams().compute);""",
        """        const half* logits = graph_->forward_token(token_id, 0, host_.streams().compute);
        return sampler_->select(logits, host_.streams().compute);""",
        True,
    ),
    (
        "EN-4 the reply is decoded with EOS still attached",
        """        std::vector<uint32_t> visible = generated.token_ids;
        if (!visible.empty() && visible.back() == tokenizer_.eos_token_id()) {
            visible.pop_back();
        }""",
        """        std::vector<uint32_t> visible = generated.token_ids;""",
        True,
    ),
    (
        "EN-5 the prompt-fits-capacity refusal is dropped",
        """        if (prompt.size() >= capacity) {
            throw std::runtime_error(
                "V4Engine::chat: the rendered prompt is " +
                std::to_string(prompt.size()) + " tokens, which leaves no room to generate "
                "inside a context of " + std::to_string(capacity));
        }""",
        """        if (false) {
            throw std::runtime_error("unreachable");
        }""",
        True,
    ),
    (
        "EN-6 strip_thinking returns the whole text",
        """        const size_t position = text.rfind(marker);
        return position == std::string::npos ? text : text.substr(position + marker.size());""",
        """        const size_t position = text.rfind(marker);
        return text;""",
        True,
    ),
    (
        "EN-7 the artifact's do_sample is ignored",
        """        config.temperature = do_sample ? temperature : 0.0f;""",
        """        config.temperature = temperature;""",
        True,
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", TARGET],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"

    env = dict(os.environ, OMP_WAIT_POLICY="passive")
    run = subprocess.run([f"./build/bin/{TARGET}"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    failures = [line.strip()
                for line in (run.stdout + run.stderr).splitlines()
                if line.strip().endswith("FAIL")]
    return run.returncode == 0, "\n".join(failures)


def main() -> int:
    original = SOURCE.read_text()
    BACKUP.write_text(original)

    survivors = []
    equivalents = []
    try:
        for name, old, new, expect_killed in MUTATIONS:
            text = BACKUP.read_text()
            count = text.count(old)
            if count != 1:
                print(f"{name}: ANCHOR MATCHED {count} TIMES — mutation not applied")
                survivors.append(name)
                continue
            SOURCE.write_text(text.replace(old, new))

            passed, detail = build_and_run()

            if not expect_killed:
                if passed:
                    equivalents.append(name)
                    print(f"{name}: equivalent (as expected — the gate is right to pass)")
                else:
                    survivors.append(name)
                    print(f"{name}: UNEXPECTEDLY KILLED — the equivalence claim is wrong")
                    for line in detail.splitlines()[:6]:
                        print(f"    {line}")
                continue

            verdict = "SURVIVED" if passed else "killed"
            if passed:
                survivors.append(name)
            print(f"{name}: {verdict}", flush=True)
            if not passed and detail != "BUILD FAILED":
                killed_by = detail.splitlines()
                for line in killed_by[:6]:
                    print(f"    {line}")
                if len(killed_by) > 6:
                    print(f"    ... and {len(killed_by) - 6} more failing lines")
    finally:
        SOURCE.write_text(original)
        subprocess.run(["cmake", "--build", "build", "--target", TARGET],
                       cwd=ROOT, capture_output=True, text=True)
        print(f"\nrestored: {SOURCE} matches the pre-sweep source: "
              f"{SOURCE.read_text() == original}")

    killable = [m for m in MUTATIONS if m[3]]
    print(f"\n{len(killable) - len(survivors)} of {len(killable)} killed")
    if equivalents:
        print(f"named equivalents (not kills): {len(equivalents)}")
    if survivors:
        print("survivors: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
