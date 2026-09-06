#!/usr/bin/env python3
"""
Project Aeon — Offline Model Formatter & Sector-Aligned Weight Serializer

Packs MoE expert weights into Aeon's native binary format (.aeon):
1. Magic Header: 'AEON' (4 bytes), format version (uint32).
2. Metadata header: num_layers, num_experts_per_layer, hidden_dim, intermediate_dim, data_type.
3. 4096-Byte Sector Padding: Enforces that the data segment and every single
   expert tensor offset starts at a strictly 4096-byte aligned boundary.
   This guarantees zero-copy Linux `io_uring` with `O_DIRECT` compatibility.
4. Generates synthetic test fixtures or converts standard weights.
"""

import argparse
import math
import struct
import sys
import numpy as np

SECTOR_SIZE = 4096  # 4KB hardware sector boundary
AEON_MAGIC = b"AEON"
AEON_FORMAT_VERSION = 1

# Supported precision types
DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_INT8 = 3

def pad_to_sector(offset: int) -> int:
    """Returns number of padding bytes required to reach the next 4096-byte boundary."""
    remainder = offset % SECTOR_SIZE
    return 0 if remainder == 0 else (SECTOR_SIZE - remainder)

def create_synthetic_moe(output_path: str,
                         num_layers: int = 1,
                         num_experts: int = 64,
                         hidden_dim: int = 2048,
                         intermediate_dim: int = 2048,
                         dtype_code: int = DTYPE_FP16):
    """
    Creates a synthetic .aeon weight file with strictly sector-aligned expert offsets.
    Each expert consists of two projection matrices (gate/up and down):
    - W_gate_up: (intermediate_dim, hidden_dim) -> (2048, 2048) in FP16 = 8 MB
    - W_down:    (hidden_dim, intermediate_dim) -> (2048, 2048) in FP16 = 8 MB
    Total unpadded weight per expert = 16 MB.
    """
    print("=" * 70)
    print("  Project Aeon — Sector-Aligned Storage Formatter")
    print("=" * 70)
    print(f"Output File        : {output_path}")
    print(f"Layers             : {num_layers}")
    print(f"Experts per Layer  : {num_experts}")
    print(f"Hidden Dimension   : {hidden_dim}")
    print(f"Intermediate Dim   : {intermediate_dim}")
    print(f"Precision          : {'FP16' if dtype_code == DTYPE_FP16 else 'Other'}")
    print(f"Sector Alignment   : {SECTOR_SIZE} bytes")
    print("-" * 70)

    # Bytes per element
    elem_size = 2 if dtype_code in (DTYPE_FP16, DTYPE_BF16) else 1
    expert_elements = (intermediate_dim * hidden_dim) + (hidden_dim * intermediate_dim)
    expert_raw_bytes = expert_elements * elem_size
    # Ensure expert size itself is a multiple of sector size
    expert_padded_bytes = math.ceil(expert_raw_bytes / SECTOR_SIZE) * SECTOR_SIZE

    print(f"Expert Raw Bytes   : {expert_raw_bytes} ({expert_raw_bytes / (1024*1024):.2f} MB)")
    print(f"Expert Aligned Size: {expert_padded_bytes} ({expert_padded_bytes / (1024*1024):.2f} MB)")

    with open(output_path, "wb") as f:
        # 1. Write Header
        # Magic (4B), Version (4B), num_layers (4B), num_experts (4B), hidden_dim (4B),
        # intermediate_dim (4B), dtype_code (4B), expert_padded_bytes (8B)
        header_fmt = "<4sIIIIIIQ"
        header_bytes = struct.pack(
            header_fmt,
            AEON_MAGIC,
            AEON_FORMAT_VERSION,
            num_layers,
            num_experts,
            hidden_dim,
            intermediate_dim,
            dtype_code,
            expert_padded_bytes
        )
        f.write(header_bytes)

        # 2. Expert Index Table: offset (uint64) and size (uint64) for each (layer, expert)
        total_expert_entries = num_layers * num_experts
        index_table_size = total_expert_entries * 16  # 8 bytes offset + 8 bytes size

        # Compute data start offset aligned to 4KB sector
        curr_offset = len(header_bytes) + index_table_size
        padding_to_data = pad_to_sector(curr_offset)
        data_start_offset = curr_offset + padding_to_data

        print(f"Index Table Size   : {index_table_size} bytes")
        print(f"Data Segment Start : 0x{data_start_offset:X} (offset {data_start_offset}, aligned: {data_start_offset % SECTOR_SIZE == 0})")

        # Build index table entries
        index_entries = []
        for i in range(total_expert_entries):
            exp_offset = data_start_offset + (i * expert_padded_bytes)
            assert exp_offset % SECTOR_SIZE == 0, f"Offset {exp_offset} not 4KB aligned!"
            index_entries.append((exp_offset, expert_padded_bytes))

        # Write index table
        for exp_offset, exp_size in index_entries:
            f.write(struct.pack("<QQ", exp_offset, exp_size))

        # Pad up to data_start_offset
        if padding_to_data > 0:
            f.write(b"\x00" * padding_to_data)

        assert f.tell() == data_start_offset, f"File position {f.tell()} != data_start_offset {data_start_offset}"

        # 3. Write Synthetic Expert Payloads
        print("Writing synthetic expert payloads...")
        np.random.seed(42)

        # Reusable deterministic expert buffer
        # In real models this loads actual weights; here we populate with valid FP16 values
        expert_data = np.random.uniform(-0.5, 0.5, size=expert_elements).astype(np.float16).tobytes()
        pad_bytes = expert_padded_bytes - expert_raw_bytes
        expert_payload = expert_data + (b"\x00" * pad_bytes)

        for l in range(num_layers):
            for e in range(num_experts):
                actual_offset = f.tell()
                expected_offset = index_entries[l * num_experts + e][0]
                assert actual_offset == expected_offset, f"Offset mismatch: {actual_offset} vs {expected_offset}"
                f.write(expert_payload)

        total_file_size = f.tell()
        print("-" * 70)
        print(f"Total File Size    : {total_file_size} bytes ({total_file_size / (1024*1024):.2f} MB)")
        print(f"Strict 4KB Aligned : {total_file_size % SECTOR_SIZE == 0}")
        print(">>> SUCCESS: .aeon model storage format written and verified! <<<")
        print("=" * 70)

def verify_aeon_file(file_path: str):
    """Verifies that an .aeon file adheres to all sector alignment rules."""
    with open(file_path, "rb") as f:
        header = f.read(36)
        magic, ver, num_layers, num_experts, h_dim, i_dim, dtype, exp_bytes = struct.unpack("<4sIIIIIIQ", header)
        assert magic == AEON_MAGIC, "Invalid magic!"
        print(f"[Verify] Magic OK, Version: {ver}, Layers: {num_layers}, Experts: {num_experts}")
        print(f"[Verify] Hidden Dim: {h_dim}, Intermediate: {i_dim}, Expert Aligned Bytes: {exp_bytes}")

        total_experts = num_layers * num_experts
        for i in range(total_experts):
            entry = f.read(16)
            off, sz = struct.unpack("<QQ", entry)
            assert off % SECTOR_SIZE == 0, f"Expert {i} offset {off} not 4KB aligned!"
            assert sz % SECTOR_SIZE == 0, f"Expert {i} size {sz} not 4KB aligned!"

    print("[Verify] All expert offsets and sizes strictly 4096-byte aligned. O_DIRECT ready!")

def main():
    parser = argparse.ArgumentParser(description="Aeon Sector-Aligned Storage Formatter")
    parser.add_argument("--output", type=str, default="/tmp/toy_moe_layer.aeon", help="Output .aeon file path")
    parser.add_argument("--layers", type=int, default=1, help="Number of MoE layers")
    parser.add_argument("--experts", type=int, default=64, help="Number of experts per layer")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension size")
    parser.add_argument("--intermediate-dim", type=int, default=2048, help="Intermediate dimension size")
    parser.add_argument("--verify-only", type=str, default=None, help="Verify existing file")

    args = parser.parse_args()

    if args.verify_only:
        verify_aeon_file(args.verify_only)
    else:
        create_synthetic_moe(
            output_path=args.output,
            num_layers=args.layers,
            num_experts=args.experts,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim
        )
        verify_aeon_file(args.output)

if __name__ == "__main__":
    main()
