#!/usr/bin/env python3
"""Mutation sweep for the production routed-expert executor.

Per the plan's second rule: for each property the gate certifies, inject the
specific wrong variant into the *implementation* and require the gate to go red.
A survivor is either a gate defect to repair or an equivalent mutation to name as
such — never something to ignore.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/architecture/deepseek_v4/core/v4_expert_executor.hpp"
BACKUP = Path("/tmp/aeon_executor_original.hpp")

MUTATIONS = [
    (
        "EX-1 the staging upload event is not awaited",
        """                CHECK_HIP(hipStreamWaitEvent(
                    streams_.compute, staging_.events[staging_idx], 0));""",
        """                if (false) {
                    CHECK_HIP(hipStreamWaitEvent(
                        streams_.compute, staging_.events[staging_idx], 0));
                }""",
    ),
    (
        "EX-2 completed transfers are never reaped",
        """        supply_.reap_registry_transfers();
        ensure_pool_headroom();""",
        """        ensure_pool_headroom();""",
    ),
    (
        "EX-3 one slot is resolved to the neighbouring expert",
        """            const int32_t slot = state_.vram_slots[static_cast<size_t>(k)];""",
        """            const int32_t slot =
                (state_.vram_slots[static_cast<size_t>(k)] + (k == 0 ? 1 : 0)) %
                static_cast<int32_t>(registry_.vram_capacity);""",
    ),
    (
        "EX-4 the shared expert is dropped from the accumulator",
        """                scratch_.d_contrib,
                V4RoutedExpertScratch::kExperts,
                moe_accum,
                moe_accum,
                V4RoutedExpertScratch::kHidden);""",
        """                scratch_.d_contrib,
                V4RoutedExpertScratch::kExperts,
                nullptr,
                moe_accum,
                V4RoutedExpertScratch::kHidden);""",
    ),
    (
        "EX-5 consumed staging slots are never returned",
        """        for (const uint32_t staging_idx : staging_in_use_) {
            staging_.release_after_gpu_transfer(staging_idx);
        }
        staging_in_use_.clear();""",
        """        /* mutated: the staging slots are never returned */""",
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", "test_v4_expert_executor"],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"
    run = subprocess.run(["./build/bin/test_v4_expert_executor"],
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
        subprocess.run(["cmake", "--build", "build", "--target", "test_v4_expert_executor"],
                       cwd=ROOT, capture_output=True, text=True)
        print(f"\nrestored: {SOURCE} matches the pre-sweep source: "
              f"{SOURCE.read_text() == original}")

    print(f"\n{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} killed")
    if survivors:
        print("survivors: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
