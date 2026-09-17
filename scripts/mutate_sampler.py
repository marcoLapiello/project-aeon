#!/usr/bin/env python3
"""Mutation sweep for the sampler and its logit-processor seam (composition plan P3).

Per the plan's second rule: for each property the gate certifies, inject the
specific wrong variant into the *implementation* and require the gate to go red.
A survivor is either a gate defect to repair or an equivalent mutation to name as
such — never something to ignore.

Every mutation here targets `core/v4_sampler.hpp`: the seam's presence and its
position in the order, the nucleus's boundary, the generator's role in the draw,
the device fast path, and the argmax's tie direction. Those are the six things the
gate's clauses are about. No mutation is a change to a kernel: the certified
argmax pair is Tier-1 territory and a kernel defect would prove nothing about this
gate.

**One named equivalent is included on purpose**, because discovering it is worth
recording. `SP-7` changes top-p's `accumulated >= top_p` to `>`: the two differ
only when the running mass lands *exactly* on the threshold, which is a
measure-zero event, so the two are behaviourally identical and the sweep must
report it as an equivalent rather than call it a survivor or pretend it was
killed. It is the same discipline as trap 42/43 — say what the instrument can and
cannot see.

`SP-5` deserves a note too. It is not a numerical change at all: it makes `select`
always take the host path. The gate catches it because the fast path is
*observable*, which is the whole reason that accessor exists — an implementation
that always widened would be correct and 259 KB slower per token, and nothing else
would notice.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/architecture/deepseek_v4/core/v4_sampler.hpp"
BACKUP = Path("/tmp/aeon_sampler_original.hpp")
TARGET = "test_v4_sampler"

# (name, old, new, expect_killed)
MUTATIONS = [
    (
        "SP-1 the seam is never invoked",
        """        if (processor_) processor_(logits, n);
        if (config_.temperature > 0.0f) {""",
        """        if (config_.temperature > 0.0f) {""",
        True,
    ),
    (
        "SP-2 the seam runs after the temperature and the truncations",
        """        if (processor_) processor_(logits, n);
        if (config_.temperature > 0.0f) {
            sampler_ops::apply_temperature(logits, n, config_.temperature);
        }
        sampler_ops::apply_top_k(logits, n, config_.top_k);
        sampler_ops::apply_top_p(logits, n, config_.top_p);""",
        """        if (config_.temperature > 0.0f) {
            sampler_ops::apply_temperature(logits, n, config_.temperature);
        }
        sampler_ops::apply_top_k(logits, n, config_.top_k);
        sampler_ops::apply_top_p(logits, n, config_.top_p);
        if (processor_) processor_(logits, n);""",
        True,
    ),
    (
        "SP-3 the nucleus drops its boundary element",
        """    for (uint32_t rank = keep; rank < n; ++rank) {
        logits[order[rank]] = -std::numeric_limits<float>::infinity();
    }""",
        """    for (uint32_t rank = (keep > 0 ? keep - 1 : 0); rank < n; ++rank) {
        logits[order[rank]] = -std::numeric_limits<float>::infinity();
    }""",
        True,
    ),
    (
        "SP-4 the draw ignores the generator and always takes the argmax",
        """        sampler_ops::softmax_in_place(logits, n);
        return sampler_ops::sample_from_probs(logits, n, rng_);""",
        """        sampler_ops::softmax_in_place(logits, n);
        (void)rng_;
        return sampler_ops::argmax_of(logits, n);""",
        True,
    ),
    (
        "SP-5 select never takes the four-byte device path",
        """        last_decision_used_device_argmax_ = (config_.temperature <= 0.0f) && !processor_;""",
        """        last_decision_used_device_argmax_ = false;""",
        True,
    ),
    (
        "SP-6 the argmax resolves a tie to the higher index",
        """        if (logits[i] > logits[best]) best = i;""",
        """        if (logits[i] >= logits[best]) best = i;""",
        True,
    ),
    (
        "SP-7 (named equivalent) top-p's boundary comparison is strict",
        """        if (accumulated >= static_cast<double>(top_p)) break;""",
        """        if (accumulated > static_cast<double>(top_p)) break;""",
        False,  # see the module docstring: measure-zero, behaviourally identical
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", TARGET],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"

    # ~8 s per run, almost all of it the one section that builds the model host.
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
