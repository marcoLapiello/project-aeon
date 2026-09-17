#!/usr/bin/env python3
"""Mutation sweep for the graph's head stage (composition plan P1).

Per the plan's second rule: for each property the gate certifies, inject the
specific wrong variant into the *implementation* and require the gate to go red.
A survivor is either a gate defect to repair or an equivalent mutation to name as
such — never something to ignore.

The mutations target `core/v4_graph.hpp`'s **wiring**, not a kernel. Tier 1 already
owns the kernels (`hc_head_wave32_kernel` has its own 6-of-6 sweep in the Step-3
gate), and what a stage gate has to be able to see is a mis-wiring between correct
kernels — which is the same split the Tier-2 layer gates used.

Any replacement that does not apply exactly once is an error, so a no-op mutation
cannot be mistaken for a survivor.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/marcolap/project-aeon")
SOURCE = ROOT / "src/architecture/deepseek_v4/core/v4_graph.hpp"
BACKUP = Path("/tmp/aeon_graph_head_original.hpp")

MUTATIONS = [
    (
        "GH-1 the HC expansion fills one stream and leaves the rest stale",
        """        for (uint32_t stream_index = 0; stream_index < hc_mult; ++stream_index) {
            CHECK_HIP(hipMemcpyAsync(
                scratch.d_res_in_half + static_cast<size_t>(stream_index) * hidden,
                row, static_cast<size_t>(hidden) * sizeof(half),
                hipMemcpyHostToDevice, stream));
        }""",
        """        CHECK_HIP(hipMemcpyAsync(
            scratch.d_res_in_half, row, static_cast<size_t>(hidden) * sizeof(half),
            hipMemcpyHostToDevice, stream));""",
    ),
    (
        "GH-2 the final RMSNorm is skipped",
        """        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel, dim3(1), dim3(32), 0, stream,
            scratch.d_hc_head_out, resources.d_final_norm, scratch.d_head_norm,
            hidden, rms_eps);""",
        """        CHECK_HIP(hipMemcpyAsync(scratch.d_head_norm, scratch.d_hc_head_out,
                                 static_cast<size_t>(hidden) * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));""",
    ),
    (
        "GH-3 the LM head reads the pre-norm vector",
        """            scratch.d_head_norm, resources.d_lm_head, scratch.d_logits, hidden);""",
        """            scratch.d_hc_head_out, resources.d_lm_head, scratch.d_logits, hidden);""",
    ),
    (
        "GH-4 hc_head is told there is one stream instead of four",
        """            scratch.d_res_in, resources.d_hc_head_fn, resources.d_hc_head_base,
            resources.d_hc_head_scale, scratch.d_hc_head_out,
            hidden, hc_mult, rms_eps, hc_eps);""",
        """            scratch.d_res_in, resources.d_hc_head_fn, resources.d_hc_head_base,
            resources.d_hc_head_scale, scratch.d_hc_head_out,
            hidden, 1, rms_eps, hc_eps);""",
    ),
    (
        "GH-5 the embedding row is not offset by the token id",
        """        const half* row = host_.resources().host_embed_table +
                          static_cast<size_t>(token_id) * hidden;""",
        """        const half* row = host_.resources().host_embed_table;""",
    ),
]


def build_and_run() -> tuple[bool, str]:
    build = subprocess.run(
        ["cmake", "--build", "build", "--target", "test_v4_graph_head"],
        cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return False, "BUILD FAILED"
    run = subprocess.run(["./build/bin/test_v4_graph_head"],
                         cwd=ROOT, capture_output=True, text=True)
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
            print(f"{name}: {verdict}")
            if not passed and detail != "BUILD FAILED":
                killed_by = detail.splitlines()
                for line in killed_by[:6]:
                    print(f"    {line}")
                if len(killed_by) > 6:
                    print(f"    ... and {len(killed_by) - 6} more failing lines")
    finally:
        SOURCE.write_text(original)
        subprocess.run(["cmake", "--build", "build", "--target", "test_v4_graph_head"],
                       cwd=ROOT, capture_output=True, text=True)
        print(f"\nrestored: {SOURCE} matches the pre-sweep source: "
              f"{SOURCE.read_text() == original}")

    print(f"\n{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} killed")
    if survivors:
        print("survivors: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
