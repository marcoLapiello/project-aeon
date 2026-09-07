#!/usr/bin/env python3
"""
Project Aeon — Surgical Safetensors to .aeon Model Converter & Sector-Aligned Serializer

Converts Hugging Face Safetensors shards of DeepSeek-V4-Flash-0731-INT4-W4A16 into
Aeon's native production format (.aeon):

Outputs created in destination directory:
1. config.json, generation_config.json, tokenizer.json, tokenizer_config.json, etc. (physically copied)
2. model_dense.aeon:
   - Header (magic 'AEON_DENSE', version, metadata, tensor directory)
   - Contiguous binary payloads of all 1,271 dense tensors (word embedding, attention projections,
     RMSNorms, Hyper-Connections Sinkhorn tables, shared experts, router gate weights, LM head).
3. model_experts.aeon:
   - 11,008 routed experts (43 layers x 256 experts)
   - Each expert is strictly 14,155,776 bytes (exactly 3,456 sectors of 4096 bytes).
   - Payload layout per expert:
     [W1_packed (4,194,304 B)] [W1_scale (524,288 B)]
     [W2_packed (4,194,304 B)] [W2_scale (524,288 B)]
     [W3_packed (4,194,304 B)] [W3_scale (524,288 B)]
   - Offset formula: slot_index = (layer_id * 256 + expert_id)
     file_offset = slot_index * 14,155,776 bytes
   - 100% compliant with Linux io_uring O_DIRECT (4096-byte sector alignment).
4. model_experts.index:
   - Compact binary lookup table: (layer_id, expert_id) -> (uint64 file_offset, uint64 byte_length)
   - Plus model architectural metadata.

Zero external dependencies: uses Python standard library only (json, struct, mmap, shutil, os, sys).
"""

import argparse
import json
import mmap
import os
import shutil
import struct
import sys
import time

SECTOR_SIZE = 4096  # 4KB hardware sector boundary
EXPERT_RAW_BYTES = 14155776  # 4MB*3 + 512KB*3 = 14,155,776 bytes (exactly 3,456 sectors)
assert EXPERT_RAW_BYTES % SECTOR_SIZE == 0

AEON_DENSE_MAGIC = b"AEON_DENSE\x00\x00"
AEON_EXP_MAGIC = b"AEON_EXPERTS\x00\x00"
AEON_FORMAT_VERSION = 1

AUX_FILES_TO_COPY = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
    ".gitattributes"
]


class SafetensorsIndex:
    """Parses Safetensors headers and maps every tensor to (shard_filename, start_byte, end_byte)."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing Safetensors index: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            self.index_data = json.load(f)

        self.weight_map = self.index_data.get("weight_map", {})
        self.shard_headers = {}
        self.shard_data_starts = {}
        self.tensor_locations = {}

        # Inspect all shards
        unique_shards = sorted(set(self.weight_map.values()))
        print(f"[Index] Found {len(unique_shards)} Safetensors shards in index.")

        for shard_name in unique_shards:
            shard_path = os.path.join(model_dir, shard_name)
            with open(shard_path, "rb") as f:
                hdr_len_bytes = f.read(8)
                hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
                hdr_json = f.read(hdr_len).decode("utf-8")
                hdr = json.loads(hdr_json)

            self.shard_headers[shard_name] = hdr
            data_start = 8 + hdr_len
            self.shard_data_starts[shard_name] = data_start

            for tensor_name, tensor_info in hdr.items():
                if tensor_name == "__metadata__":
                    continue
                d_offsets = tensor_info["data_offsets"]
                start = data_start + d_offsets[0]
                end = data_start + d_offsets[1]
                size = end - start
                self.tensor_locations[tensor_name] = {
                    "shard": shard_name,
                    "start": start,
                    "end": end,
                    "size": size,
                    "dtype": tensor_info["dtype"],
                    "shape": tensor_info["shape"]
                }


class ShardMmapPool:
    """Provides memory-mapped access to Safetensors shards on demand."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.open_files = {}
        self.mmaps = {}

    def get_slice(self, shard_name: str, start: int, end: int) -> memoryview:
        if shard_name not in self.mmaps:
            shard_path = os.path.join(self.model_dir, shard_name)
            f = open(shard_path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.open_files[shard_name] = f
            self.mmaps[shard_name] = mm
        return memoryview(self.mmaps[shard_name])[start:end]

    def close(self):
        for mm in self.mmaps.values():
            mm.close()
        for f in self.open_files.values():
            f.close()
        self.mmaps.clear()
        self.open_files.clear()


def copy_auxiliary_files(source_dir: str, output_dir: str):
    """Physically copies configuration, tokenizer, and license files without symlinking."""
    print("\n" + "=" * 70)
    print("  Step 1: Physically Copying Auxiliary Configuration & Tokenizer Files")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    copied_count = 0
    for filename in AUX_FILES_TO_COPY:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            print(f"  - Copying: {filename} -> {output_dir}")
            # Ensure physical copy, not symlink
            if os.path.islink(src):
                real_src = os.path.realpath(src)
                shutil.copyfile(real_src, dst)
            else:
                shutil.copyfile(src, dst)
            copied_count += 1
        else:
            print(f"  - Notice: {filename} not found, skipping.")
    print(f"[Done] Copied {copied_count} files successfully.")


def convert_dense_tensors(source_index: SafetensorsIndex,
                          mmap_pool: ShardMmapPool,
                          output_dir: str,
                          max_layers: int = None):
    """
    Extracts all non-routed-expert weights (attention, norms, shared experts, router, head)
    into model_dense.aeon.
    """
    print("\n" + "=" * 70)
    print("  Step 2: Serializing Dense Backbone -> model_dense.aeon")
    print("=" * 70)

    dense_keys = [
        k for k in sorted(source_index.tensor_locations.keys())
        if ("experts." not in k or "shared_experts" in k)
    ]

    # If limited layers requested (e.g. for quick test)
    if max_layers is not None:
        filtered = []
        for k in dense_keys:
            if k.startswith("layers."):
                parts = k.split(".")
                lid = int(parts[1])
                if lid < max_layers:
                    filtered.append(k)
            else:
                filtered.append(k)
        dense_keys = filtered

    print(f"  - Total dense tensors to serialize: {len(dense_keys)}")

    output_dense_path = os.path.join(output_dir, "model_dense.aeon")
    
    # Calculate directory table and total payload size
    dense_directory = []
    current_data_offset = 0

    for k in dense_keys:
        loc = source_index.tensor_locations[k]
        dense_directory.append({
            "name": k,
            "offset": current_data_offset,
            "size": loc["size"],
            "dtype": loc["dtype"],
            "shape": loc["shape"]
        })
        current_data_offset += loc["size"]

    dir_json_bytes = json.dumps(dense_directory, indent=2).encode("utf-8")
    header_fmt = "<12sIIQ"
    header_magic = AEON_DENSE_MAGIC
    header_version = AEON_FORMAT_VERSION
    header_tensor_count = len(dense_keys)
    header_dir_len = len(dir_json_bytes)

    header_bytes = struct.pack(header_fmt, header_magic, header_version, header_tensor_count, header_dir_len)

    # 4KB align the start of raw tensor data
    pre_data_len = len(header_bytes) + header_dir_len
    pad_bytes = (SECTOR_SIZE - (pre_data_len % SECTOR_SIZE)) % SECTOR_SIZE
    data_start = pre_data_len + pad_bytes

    print(f"  - Header + Directory: {pre_data_len} bytes (padding to 4KB: {pad_bytes} bytes)")
    print(f"  - Raw Tensor Payload: {current_data_offset / (1024**3):.2f} GB")
    print(f"  - Target file: {output_dense_path}")

    start_time = time.time()
    with open(output_dense_path, "wb") as f:
        f.write(header_bytes)
        f.write(dir_json_bytes)
        if pad_bytes > 0:
            f.write(b"\x00" * pad_bytes)

        written_bytes = 0
        for idx, entry in enumerate(dense_directory):
            k = entry["name"]
            loc = source_index.tensor_locations[k]
            tensor_slice = mmap_pool.get_slice(loc["shard"], loc["start"], loc["end"])
            f.write(tensor_slice)
            written_bytes += loc["size"]
            if (idx + 1) % 200 == 0 or (idx + 1) == len(dense_directory):
                elapsed = time.time() - start_time
                mb_s = (written_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                print(f"    Written {idx + 1}/{len(dense_directory)} tensors ({written_bytes / (1024**3):.2f} GB, {mb_s:.1f} MB/s)")

    total_dense_size = os.path.getsize(output_dense_path)
    print(f"[Done] model_dense.aeon created: {total_dense_size / (1024**3):.2f} GB in {time.time() - start_time:.1f}s")


def convert_routed_experts(source_index: SafetensorsIndex,
                           mmap_pool: ShardMmapPool,
                           output_dir: str,
                           max_layers: int = None):
    """
    Extracts all 11,008 routed experts into model_experts.aeon and generates model_experts.index.
    Every expert is laid out as a contiguous 14,155,776-byte block (strictly 4KB sector aligned).
    """
    total_layers = 43 if max_layers is None else max_layers
    experts_per_layer = 256
    total_experts = total_layers * experts_per_layer

    print("\n" + "=" * 70)
    print(f"  Step 3: Serializing Routed Experts ({total_layers} layers x 256 experts = {total_experts} total)")
    print("=" * 70)

    output_experts_path = os.path.join(output_dir, "model_experts.aeon")
    output_index_path = os.path.join(output_dir, "model_experts.index")

    total_experts_bytes = total_experts * EXPERT_RAW_BYTES
    print(f"  - Total Payload Size: {total_experts_bytes / (1024**3):.2f} GB")
    print(f"  - Payload per Expert: {EXPERT_RAW_BYTES} bytes (3,456 sectors of 4KB)")
    print(f"  - Target file       : {output_experts_path}")

    # Binary index table format:
    # Header: Magic b'AEON_EXPERTS\0\0' (12B), Version (4B), Layers (4B), ExpertsPerLayer (4B), ExpertBytes (8B)
    index_header_fmt = "<12sIIIQ"
    index_header = struct.pack(
        index_header_fmt,
        AEON_EXP_MAGIC,
        AEON_FORMAT_VERSION,
        total_layers,
        experts_per_layer,
        EXPERT_RAW_BYTES
    )

    index_entries = []
    for slot_idx in range(total_experts):
        offset = slot_idx * EXPERT_RAW_BYTES
        assert offset % SECTOR_SIZE == 0, f"Offset {offset} is not 4KB aligned!"
        index_entries.append((offset, EXPERT_RAW_BYTES))

    print(f"  - Writing index table to {output_index_path}...")
    with open(output_index_path, "wb") as f_idx:
        f_idx.write(index_header)
        for off, sz in index_entries:
            f_idx.write(struct.pack("<QQ", off, sz))
    print(f"  - Index table written ({os.path.getsize(output_index_path)} bytes).")

    # Serializing experts into model_experts.aeon
    start_time = time.time()
    tensors_order = [
        ("w1", "weight_packed"),
        ("w1", "weight_scale"),
        ("w2", "weight_packed"),
        ("w2", "weight_scale"),
        ("w3", "weight_packed"),
        ("w3", "weight_scale")
    ]

    with open(output_experts_path, "wb") as f_exp:
        processed_experts = 0
        bytes_written = 0

        for l in range(total_layers):
            layer_start = time.time()
            for e in range(experts_per_layer):
                expert_bytes = bytearray()
                for w, t in tensors_order:
                    key = f"layers.{l}.ffn.experts.{e}.{w}.{t}"
                    if key not in source_index.tensor_locations:
                        raise KeyError(f"Missing tensor in Safetensors: {key}")
                    loc = source_index.tensor_locations[key]
                    chunk = mmap_pool.get_slice(loc["shard"], loc["start"], loc["end"])
                    expert_bytes.extend(chunk)

                if len(expert_bytes) != EXPERT_RAW_BYTES:
                    raise ValueError(f"Expert L{l} E{e} size mismatch: got {len(expert_bytes)}, expected {EXPERT_RAW_BYTES}")

                f_exp.write(expert_bytes)
                processed_experts += 1
                bytes_written += len(expert_bytes)

            layer_elapsed = time.time() - layer_start
            total_elapsed = time.time() - start_time
            layer_mb_s = (experts_per_layer * EXPERT_RAW_BYTES / (1024 * 1024)) / max(layer_elapsed, 0.001)
            overall_gb = bytes_written / (1024**3)
            print(f"    [Layer {l:02d}/{total_layers-1:02d} done] 256 experts serialized ({layer_mb_s:.1f} MB/s) | Total: {overall_gb:.2f} GB ({total_elapsed:.1f}s)")

    print(f"[Done] model_experts.aeon created: {os.path.getsize(output_experts_path) / (1024**3):.2f} GB in {time.time() - start_time:.1f}s")


def verify_conversion(source_index: SafetensorsIndex,
                      mmap_pool: ShardMmapPool,
                      output_dir: str,
                      max_layers: int = None):
    """
    Performs bit-exact verification comparing sample converted tensors against original Safetensors.
    """
    print("\n" + "=" * 70)
    print("  Step 4: Bit-Exact Numerical Parity Verification")
    print("=" * 70)

    # 1. Verify model_dense.aeon
    dense_path = os.path.join(output_dir, "model_dense.aeon")
    with open(dense_path, "rb") as f:
        hdr = f.read(28)
        magic, ver, count, dir_len = struct.unpack("<12sIIQ", hdr)
        assert magic == AEON_DENSE_MAGIC, "Invalid dense magic!"
        dir_json = f.read(dir_len).decode("utf-8")
        dense_dir = json.loads(dir_json)
        
        pre_data_len = 28 + dir_len
        pad_bytes = (SECTOR_SIZE - (pre_data_len % SECTOR_SIZE)) % SECTOR_SIZE
        data_start = pre_data_len + pad_bytes

        print(f"  [Verify Dense] Format OK: {count} tensors. Checking 10 random tensors...")
        import random
        random.seed(42)
        sample_entries = random.sample(dense_dir, min(10, len(dense_dir)))
        for entry in sample_entries:
            f.seek(data_start + entry["offset"])
            converted_data = f.read(entry["size"])
            loc = source_index.tensor_locations[entry["name"]]
            source_slice = bytes(mmap_pool.get_slice(loc["shard"], loc["start"], loc["end"]))
            assert converted_data == source_slice, f"Mismatch in dense tensor: {entry['name']}"
        print("  [Verify Dense] All sampled dense tensors are 100% BIT-EXACT to source Safetensors!")

    # 2. Verify model_experts.aeon
    experts_path = os.path.join(output_dir, "model_experts.aeon")
    total_layers = 43 if max_layers is None else max_layers
    with open(experts_path, "rb") as f:
        print(f"  [Verify Experts] Checking sample routed experts across {total_layers} layers...")
        sample_experts = [(0, 0), (0, 255), (total_layers - 1, 0), (total_layers - 1, 128)]
        for l, e in sample_experts:
            slot_idx = l * 256 + e
            f.seek(slot_idx * EXPERT_RAW_BYTES)
            exp_data = f.read(EXPERT_RAW_BYTES)

            # Reconstruct from source
            expected = bytearray()
            for w, t in [("w1", "weight_packed"), ("w1", "weight_scale"),
                         ("w2", "weight_packed"), ("w2", "weight_scale"),
                         ("w3", "weight_packed"), ("w3", "weight_scale")]:
                loc = source_index.tensor_locations[f"layers.{l}.ffn.experts.{e}.{w}.{t}"]
                expected.extend(mmap_pool.get_slice(loc["shard"], loc["start"], loc["end"]))

            assert exp_data == bytes(expected), f"Mismatch in expert L{l} E{e}!"
            print(f"    - Layer {l}, Expert {e}: BIT-EXACT match ({len(exp_data)} bytes verified).")

    print("  [Verify Experts] All verified experts are 100% BIT-EXACT. Sector alignment confirmed!")
    print("\n>>> ALL CHECKS PASSED: MODEL SUCCESSFULLY CONVERTED AND VERIFIED <<<")


def main():
    parser = argparse.ArgumentParser(description="Convert Safetensors to Aeon Format")
    parser.add_argument("--model-dir", type=str,
                        default="models/DeepSeek-V4-Flash-0731-INT4-W4A16/models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0",
                        help="Path to source Safetensors snapshot")
    parser.add_argument("--output-dir", type=str,
                        default="models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon",
                        help="Path to destination Aeon model directory")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Limit conversion to first N layers (for quick testing, e.g. --max-layers 1)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Run verification on existing output directory")

    args = parser.parse_args()

    source_index = SafetensorsIndex(args.model_dir)
    mmap_pool = ShardMmapPool(args.model_dir)

    try:
        if args.verify_only:
            verify_conversion(source_index, mmap_pool, args.output_dir, args.max_layers)
        else:
            copy_auxiliary_files(args.model_dir, args.output_dir)
            convert_dense_tensors(source_index, mmap_pool, args.output_dir, args.max_layers)
            convert_routed_experts(source_index, mmap_pool, args.output_dir, args.max_layers)
            verify_conversion(source_index, mmap_pool, args.output_dir, args.max_layers)
    finally:
        mmap_pool.close()


if __name__ == "__main__":
    main()
