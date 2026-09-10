#!/usr/bin/env python3
"""Create the parallel kernel-oriented Aeon expert artifact.

The existing model_experts.aeon file is left untouched. This repacker reads
one expert at a time, applies the W1/W3 or W2 swizzle, and writes a second raw
expert file with the same per-expert byte count and sector alignment.

The companion index uses version 2 to identify the swizzled payload. The
current runtime continues to use the version-1 model_experts.aeon artifact.
"""

import argparse
import mmap
import os
import struct
import time

import numpy as np


SECTOR_SIZE = 4096
EXPERT_RAW_BYTES = 14155776
INDEX_HEADER = struct.Struct("<12sIIIQ")
INDEX_ENTRY = struct.Struct("<QQ")
INDEX_MAGIC = b"AEON_EXPERTS"
SWIZZLED_INDEX_VERSION = 2

W1_PACKED_OFFSET = 0
W1_SCALE_OFFSET = 4194304
W2_PACKED_OFFSET = 4718592
W2_SCALE_OFFSET = 8912896
W3_PACKED_OFFSET = 9437184
W3_SCALE_OFFSET = 13631488

NIBBLE_PERM = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
INVERSE_NIBBLE_PERM = np.argsort(NIBBLE_PERM)

TENSORS = (
    ("w1", W1_PACKED_OFFSET, 2048, 4096, 4, 8, np.dtype("<u4")),
    ("w1_scale", W1_SCALE_OFFSET, 2048, 4096, 4, 8, np.dtype("<f2")),
    ("w2", W2_PACKED_OFFSET, 4096, 2048, 8, 4, np.dtype("<u4")),
    ("w2_scale", W2_SCALE_OFFSET, 4096, 2048, 8, 4, np.dtype("<f2")),
    ("w3", W3_PACKED_OFFSET, 2048, 4096, 4, 8, np.dtype("<u4")),
    ("w3_scale", W3_SCALE_OFFSET, 2048, 4096, 4, 8, np.dtype("<f2")),
)


def permute_words(words: np.ndarray) -> np.ndarray:
    result = np.zeros_like(words)
    for destination_position, source_position in enumerate(NIBBLE_PERM):
        result |= (
            (words >> np.uint32(4 * int(source_position))) & np.uint32(0xF)
        ) << np.uint32(4 * destination_position)
    return result


def unpermute_words(words: np.ndarray) -> np.ndarray:
    result = np.zeros_like(words)
    for destination_position, source_position in enumerate(INVERSE_NIBBLE_PERM):
        result |= (
            (words >> np.uint32(4 * int(source_position))) & np.uint32(0xF)
        ) << np.uint32(4 * destination_position)
    return result


def swizzle_tensor(source: np.ndarray, rpw: int, lpr: int) -> np.ndarray:
    rows, columns = source.shape
    inner = 4 if source.dtype == np.dtype("<u4") else 1
    groups = columns // inner
    iterations = groups // lpr
    row_blocks = rows // rpw
    reshaped = source.reshape(row_blocks, rpw, iterations, lpr, inner)
    return np.ascontiguousarray(reshaped.transpose(0, 2, 1, 3, 4)).reshape(-1)


def unswizzle_tensor(source: np.ndarray, rows: int, columns: int, rpw: int, lpr: int) -> np.ndarray:
    inner = 4 if source.dtype == np.dtype("<u4") else 1
    groups = columns // inner
    iterations = groups // lpr
    row_blocks = rows // rpw
    reshaped = source.reshape(row_blocks, iterations, rpw, lpr, inner)
    return np.ascontiguousarray(reshaped.transpose(0, 2, 1, 3, 4)).reshape(rows, columns)


def swizzle_packed(source: np.ndarray, rows: int, K: int, rpw: int, lpr: int) -> bytes:
    permuted = permute_words(source.reshape(rows, K // 8))
    return swizzle_tensor(permuted, rpw, lpr).tobytes()


def swizzle_scale(source: np.ndarray, rows: int, K: int, rpw: int, lpr: int) -> bytes:
    shaped = source.reshape(rows, K // 32)
    return swizzle_tensor(shaped, rpw, lpr).tobytes()


def read_index(index_path: str) -> tuple[int, int, int, list[tuple[int, int]]]:
    with open(index_path, "rb") as index_file:
        header = index_file.read(INDEX_HEADER.size)
        if len(header) != INDEX_HEADER.size:
            raise ValueError(f"Index is too small: {index_path}")
        magic, version, layers, experts_per_layer, expert_bytes = INDEX_HEADER.unpack(header)
        if magic != INDEX_MAGIC:
            raise ValueError(f"Unexpected expert index magic: {magic!r}")
        if expert_bytes != EXPERT_RAW_BYTES:
            raise ValueError(
                f"Unexpected expert byte count: {expert_bytes}, expected {EXPERT_RAW_BYTES}"
            )
        total_experts = layers * experts_per_layer
        entries = [
            INDEX_ENTRY.unpack(index_file.read(INDEX_ENTRY.size))
            for _ in range(total_experts)
        ]
        if any(len(entry) != 2 for entry in entries):
            raise ValueError(f"Incomplete expert index: {index_path}")
        return version, layers, experts_per_layer, entries


def validate_source(input_path: str, index_path: str):
    version, layers, experts_per_layer, entries = read_index(index_path)
    input_size = os.path.getsize(input_path)
    if input_size < EXPERT_RAW_BYTES * layers * experts_per_layer:
        raise ValueError("Expert payload is smaller than its index describes")
    for slot, (offset, size) in enumerate(entries):
        if offset % SECTOR_SIZE != 0 or size != EXPERT_RAW_BYTES:
            raise ValueError(f"Invalid source entry {slot}: offset={offset}, size={size}")
        if offset + size > input_size:
            raise ValueError(f"Source entry {slot} exceeds the expert payload")
    return version, layers, experts_per_layer, entries


def repack_expert(source_map: mmap.mmap, source_offset: int) -> bytes:
    chunks = []
    for name, tensor_offset, rows, K, rpw, lpr, dtype in TENSORS:
        del name
        if dtype == np.dtype("<u4"):
            count = rows * (K // 8)
            source = np.frombuffer(source_map, dtype=dtype, count=count,
                                   offset=source_offset + tensor_offset).copy()
            chunks.append(swizzle_packed(source, rows, K, rpw, lpr))
        else:
            count = rows * (K // 32)
            source = np.frombuffer(source_map, dtype=dtype, count=count,
                                   offset=source_offset + tensor_offset).copy()
            chunks.append(swizzle_scale(source, rows, K, rpw, lpr))
    payload = b"".join(chunks)
    if len(payload) != EXPERT_RAW_BYTES:
        raise ValueError(f"Swizzled expert size mismatch: {len(payload)}")
    return payload


def parse_expert_spec(spec: str, layers: int, experts_per_layer: int) -> tuple[int, int]:
    try:
        layer_text, expert_text = spec.split(":", 1)
        layer = int(layer_text)
        expert = int(expert_text)
    except ValueError as error:
        raise ValueError(f"Invalid expert specification {spec!r}; expected L:E") from error
    if not (0 <= layer < layers and 0 <= expert < experts_per_layer):
        raise ValueError(f"Expert {spec!r} is outside the output dimensions")
    return layer, expert


def verify_experts(
    source_map: mmap.mmap,
    output_map: mmap.mmap,
    source_entries: list[tuple[int, int]],
    output_layers: int,
    output_experts_per_layer: int,
    source_experts_per_layer: int,
    specs: list[str],
):
    if not specs:
        specs = ["0:0", f"{output_layers - 1}:{output_experts_per_layer - 1}"]
    for spec in specs:
        layer, expert = parse_expert_spec(spec, output_layers, output_experts_per_layer)
        source_slot = layer * source_experts_per_layer + expert
        output_slot = layer * output_experts_per_layer + expert
        source_offset = source_entries[source_slot][0]
        output_offset = output_slot * EXPERT_RAW_BYTES
        for name, tensor_offset, rows, K, rpw, lpr, dtype in TENSORS:
            del name
            if dtype == np.dtype("<u4"):
                count = rows * (K // 8)
                source = np.frombuffer(source_map, dtype=dtype, count=count,
                                                                             offset=source_offset + tensor_offset).copy().reshape(rows, K // 8)
                swizzled = np.frombuffer(output_map, dtype=dtype, count=count,
                                                                                 offset=output_offset + tensor_offset).copy()
                restored = unpermute_words(
                    unswizzle_tensor(swizzled, rows, K // 8, rpw, lpr),
                )
                if not np.array_equal(source, restored):
                    raise AssertionError(f"Packed tensor mismatch for expert {spec}")
            else:
                count = rows * (K // 32)
                source = np.frombuffer(source_map, dtype=dtype, count=count,
                                                                             offset=source_offset + tensor_offset).copy().reshape(rows, K // 32)
                swizzled = np.frombuffer(output_map, dtype=dtype, count=count,
                                                                                 offset=output_offset + tensor_offset).copy()
                restored = unswizzle_tensor(swizzled, rows, K // 32, rpw, lpr)
                if not np.array_equal(source, restored):
                    raise AssertionError(f"Scale tensor mismatch for expert {spec}")
        print(f"[PASS] Real expert {spec} round-trips bit-identically")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-experts", required=True,
                        help="Existing version-1 model_experts.aeon")
    parser.add_argument("--input-index", required=True,
                        help="Existing version-1 model_experts.index")
    parser.add_argument("--output-experts", required=True,
                        help="New swizzled raw expert payload")
    parser.add_argument("--output-index", required=True,
                        help="New version-2 swizzled expert index")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Repack only a prefix of layers")
    parser.add_argument("--max-experts-per-layer", type=int, default=None,
                        help="Repack only a prefix of experts in each layer")
    parser.add_argument("--verify-expert", action="append", default=[], metavar="L:E",
                        help="Verify a selected output expert; may be repeated")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing output paths after successful repacking")
    args = parser.parse_args()

    input_experts = os.path.abspath(args.input_experts)
    input_index = os.path.abspath(args.input_index)
    output_experts = os.path.abspath(args.output_experts)
    output_index = os.path.abspath(args.output_index)
    if input_experts == output_experts or input_index == output_index:
        raise ValueError("Input and output paths must be different")
    if not args.force and (os.path.exists(output_experts) or os.path.exists(output_index)):
        raise FileExistsError("Output exists; pass --force to replace it")

    source_version, source_layers, source_experts_per_layer, source_entries = validate_source(
        input_experts, input_index)
    if source_version != 1:
        raise ValueError(f"Expected version-1 source index, got version {source_version}")

    output_layers = source_layers if args.max_layers is None else args.max_layers
    output_experts_per_layer = (
        source_experts_per_layer if args.max_experts_per_layer is None
        else args.max_experts_per_layer
    )
    if not (0 < output_layers <= source_layers):
        raise ValueError("--max-layers must be between 1 and the source layer count")
    if not (0 < output_experts_per_layer <= source_experts_per_layer):
        raise ValueError(
            "--max-experts-per-layer must be between 1 and the source expert count"
        )

    os.makedirs(os.path.dirname(output_experts) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_index) or ".", exist_ok=True)
    output_experts_tmp = output_experts + ".tmp"
    output_index_tmp = output_index + ".tmp"
    start_time = time.perf_counter()
    total_output_experts = output_layers * output_experts_per_layer

    try:
        with open(input_experts, "rb") as input_file:
            source_map = mmap.mmap(input_file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                with open(output_experts_tmp, "wb") as output_file:
                    processed = 0
                    for layer in range(output_layers):
                        for expert in range(output_experts_per_layer):
                            source_slot = layer * source_experts_per_layer + expert
                            payload = repack_expert(source_map, source_entries[source_slot][0])
                            output_file.write(payload)
                            processed += 1
                        elapsed = max(time.perf_counter() - start_time, 0.001)
                        rate = processed * EXPERT_RAW_BYTES / (1024 ** 2) / elapsed
                        print(f"[Layer {layer:02d}/{output_layers - 1:02d}] "
                              f"{processed}/{total_output_experts} experts, {rate:.1f} MiB/s")
                    output_file.flush()
                    os.fsync(output_file.fileno())
            finally:
                source_map.close()

        with open(output_index_tmp, "wb") as index_file:
            index_file.write(INDEX_HEADER.pack(
                INDEX_MAGIC, SWIZZLED_INDEX_VERSION,
                output_layers, output_experts_per_layer, EXPERT_RAW_BYTES))
            for slot in range(total_output_experts):
                offset = slot * EXPERT_RAW_BYTES
                index_file.write(INDEX_ENTRY.pack(offset, EXPERT_RAW_BYTES))
            index_file.flush()
            os.fsync(index_file.fileno())

        os.replace(output_experts_tmp, output_experts)
        os.replace(output_index_tmp, output_index)
    except Exception:
        for temporary_path in (output_experts_tmp, output_index_tmp):
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
        raise

    with open(input_experts, "rb") as input_file, open(output_experts, "rb") as output_file:
        source_map = mmap.mmap(input_file.fileno(), 0, access=mmap.ACCESS_READ)
        output_map = mmap.mmap(output_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            verify_experts(source_map, output_map, source_entries,
                           output_layers, output_experts_per_layer,
                           source_experts_per_layer, args.verify_expert)
        finally:
            source_map.close()
            output_map.close()

    elapsed = time.perf_counter() - start_time
    print(f"[Done] Wrote {total_output_experts} swizzled experts in {elapsed:.1f}s")
    print(f"       Payload: {output_experts}")
    print(f"       Index:   {output_index} (version {SWIZZLED_INDEX_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())