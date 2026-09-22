#!/usr/bin/env python3
"""Mutation sweep for the frozen-prefill policy (Step 6 D-b).

Per the plan's second rule: for each property the outcome-3 gate certifies, inject
the specific wrong variant into the *implementation* and require the gate to go
red. The gate asserts two things that must be true together — Warm is unchanged,
and the prefill really read Warm — so each has a mutation that breaks exactly it.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/infrastructure/core/expert_registry.hpp"
BACKUP = Path("/tmp/aeon_registry_original.hpp")

MUTATIONS = [
    (
        "WF-1 the frozen Warm request promotes normally (drains Warm)",
        """        if (warm_frozen_ && entry.owner == ExpertTier::WARM_HOST) {""",
        """        if (false && entry.owner == ExpertTier::WARM_HOST) {""",
    ),
    (
        "WF-2 the freeze never engages",
        """        warm_frozen_ = frozen;
        if (!frozen) {
            release_shadow_residencies();
        }""",
        """        warm_frozen_ = false;
        if (!frozen) {
            release_shadow_residencies();
        }""",
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", "test_v4_warm_frozen_prefill"],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"
    run = subprocess.run(["./build/bin/test_v4_warm_frozen_prefill"],
                         cwd=ROOT, capture_output=True, text=True)
    tail = (run.stdout + run.stderr).strip().splitlines()
    return run.returncode == 0, "\n".join(tail[-3:])


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
            print(f"{name}: {verdict}")
            if not passed and detail != "BUILD FAILED":
                for line in detail.splitlines():
                    if "failed" in line or "FAIL" in line:
                        print(f"    {line.strip()}")
    finally:
        SOURCE.write_text(original)
        subprocess.run(
            ["cmake", "--build", "build", "--target", "test_v4_warm_frozen_prefill"],
            cwd=ROOT, capture_output=True, text=True)
        print(f"\nrestored: {SOURCE} matches the pre-sweep source: "
              f"{SOURCE.read_text() == original}")

    print(f"\n{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} killed")
    if survivors:
        print("survivors: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
