#!/usr/bin/env python3
"""Mutation sweep for the 43-layer driver (composition plan P2).

Per the plan's second rule: for each property the gate certifies, inject the
specific wrong variant into the *implementation* and require the gate to go red.
A survivor is either a gate defect to repair or an equivalent mutation to name as
such — never something to ignore.

Every mutation here targets `core/v4_graph.hpp`'s **driver wiring** — the loop, the
embedding, the head's position in the sequence — because that is what P2 adds.
The layer body is Tier 2's and the kernels are Tier 1's; a phase gate has to be
able to see a mis-composition of correct parts, which is the same split the Tier-2
layer gates and the P1 sweep used.

No mutation here is a numerical change to a kernel, and that is deliberate: a
kernel defect is the wrong instrument for a driver sweep, so it would prove
nothing about this gate.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/architecture/deepseek_v4/core/v4_graph.hpp"
BACKUP = Path("/tmp/aeon_graph_body_original.hpp")
TARGET = "test_v4_graph_body"

MUTATIONS = [
    (
        "GB-1 forward_token runs the first 42 layers, not 43",
        """        const uint32_t layers = host_.num_layers();
        for (uint32_t layer = 0; layer < layers; ++layer) {
            (void)run_layer(layer, token_id, position, stream);
        }""",
        """        const uint32_t layers = host_.num_layers();
        for (uint32_t layer = 0; layer + 1 < layers; ++layer) {
            (void)run_layer(layer, token_id, position, stream);
        }""",
    ),
    (
        "GB-2 run_layer reads layer 0 whatever it is asked for",
        """        return run_layer_body_decoding(
            host_.layer(layer_id), host_.scratch(), tables, token_id, position, stream,
            host_.executor(), observer_);""",
        """        return run_layer_body_decoding(
            host_.layer(0), host_.scratch(), tables, token_id, position, stream,
            host_.executor(), observer_);""",
    ),
    (
        "GB-3 forward_token drops the embedding",
        """        embed_token(token_id, stream);

        const uint32_t layers = host_.num_layers();""",
        """        const uint32_t layers = host_.num_layers();""",
    ),
    (
        "GB-4 forward_token runs the head before the layers",
        """        const uint32_t layers = host_.num_layers();
        for (uint32_t layer = 0; layer < layers; ++layer) {
            (void)run_layer(layer, token_id, position, stream);
        }

        const half* logits = head_stage(stream);""",
        """        const uint32_t layers = host_.num_layers();
        const half* logits = head_stage(stream);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            (void)run_layer(layer, token_id, position, stream);
        }""",
    ),
    (
        "GB-5 run_layer hands every layer position 0",
        """            host_.layer(layer_id), host_.scratch(), tables, token_id, position, stream,
            host_.executor(), observer_);""",
        """            host_.layer(layer_id), host_.scratch(), tables, token_id, 0, stream,
            host_.executor(), observer_);""",
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", TARGET],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"

    # The gate is the model forward pass: ~3 minutes per run, and its reference is
    # OpenMP-parallel, so passive waiting keeps the sweep from burning every core
    # it is not using.
    env = dict(os.environ, OMP_WAIT_POLICY="passive")
    run = subprocess.run([f"./build/bin/{TARGET}"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    # Every failing line, so each mutation is recorded with the checks it killed
    # rather than only a count.
    failures = [line.strip()
                for line in (run.stdout + run.stderr).splitlines()
                if line.strip().endswith("FAIL")]
    return run.returncode == 0, "\n".join(failures)


def main() -> int:
    original = SOURCE.read_text()
    BACKUP.write_text(original)

    survivors = []
    try:
        for name, old, new in MUTATIONS:
            text = BACKUP.read_text()
            count = text.count(old)
            if count != 1:
                print(f"{name}: ANCHOR MATCHED {count} TIMES — mutation not applied")
                survivors.append(name)
                continue
            SOURCE.write_text(text.replace(old, new))

            passed, detail = build_and_run()
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

    print(f"\n{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} killed")
    if survivors:
        print("survivors: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
