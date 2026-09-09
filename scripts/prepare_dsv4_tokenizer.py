#!/usr/bin/env python3
"""Prepare the DSV4 tokenizer JSON for the native runtime."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"AEONTOK1"
VERSION = 2


def write_u32(output, value):
    output.write(struct.pack("<I", value))


def write_string(output, value):
    encoded = value.encode("utf-8")
    write_u32(output, len(encoded))
    output.write(encoded)


def parse_merge(merge):
    if isinstance(merge, list):
        if len(merge) != 2:
            raise ValueError(f"Invalid merge entry: {merge!r}")
        return merge[0], merge[1]
    left, separator, right = merge.partition(" ")
    if not separator or not left or not right:
        raise ValueError(f"Invalid merge entry: {merge!r}")
    return left, right


def prepare(source_path, output_path):
    source_bytes = source_path.read_bytes()
    tokenizer = json.loads(source_bytes)
    model = tokenizer.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Expected a BPE tokenizer, got {model.get('type')!r}")

    vocab = model["vocab"]
    vocab_by_id = {}
    for token, token_id in vocab.items():
        if token_id in vocab_by_id:
            raise ValueError(f"Duplicate base vocabulary ID {token_id}")
        vocab_by_id[token_id] = token

    added_tokens = tokenizer.get("added_tokens", [])
    added_by_id = {}
    for token in added_tokens:
        token_id = token["id"]
        if token_id in added_by_id:
            raise ValueError(f"Duplicate added-token ID {token_id}")
        added_by_id[token_id] = token

    all_ids = set(vocab_by_id) | set(added_by_id)
    if not all_ids or min(all_ids) != 0:
        raise ValueError("Tokenizer IDs must start at zero")
    id_limit = max(all_ids) + 1
    base_vocab_size = len(vocab_by_id)

    special = {token["content"]: token["id"] for token in added_tokens}
    required_tokens = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
    ]
    required_ids = [special.get(content) for content in required_tokens]
    if any(token_id is None for token_id in required_ids):
        raise ValueError("The tokenizer is missing a required DSV4 control token")

    merges = [parse_merge(merge) for merge in model["merges"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(MAGIC)
        output.write(
            struct.pack(
                "<IIIIIIIIIII",
                VERSION,
                base_vocab_size,
                id_limit,
                len(merges),
                len(added_tokens),
                *required_ids,
            )
        )
        output.write(hashlib.sha256(source_bytes).digest())

        for token_id, token in sorted(vocab_by_id.items()):
            write_u32(output, token_id)
            write_string(output, token)

        for left, right in merges:
            write_string(output, left)
            write_string(output, right)

        for token in sorted(added_tokens, key=lambda item: item["id"]):
            flags = 0
            flags |= int(token.get("special", False)) << 0
            flags |= int(token.get("normalized", False)) << 1
            flags |= int(token.get("single_word", False)) << 2
            flags |= int(token.get("lstrip", False)) << 3
            flags |= int(token.get("rstrip", False)) << 4
            write_u32(output, token["id"])
            output.write(struct.pack("<B", flags))
            write_string(output, token["content"])

    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")
    print(f"base vocabulary: {base_vocab_size}, ID limit: {id_limit}")
    print(f"merges: {len(merges)}, added tokens: {len(added_tokens)}")
    print(f"source sha256: {hashlib.sha256(source_bytes).hexdigest()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()