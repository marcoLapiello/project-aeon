#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${1:-"$repo_root/models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon"}
output_path=${2:-"$repo_root/build/correctness/stage0_model_contract.json"}
contract_tool=${AEON_MODEL_CONTRACT_TOOL:-"$repo_root/build/bin/aeon_model_contract"}

if [[ ! -x "$contract_tool" ]]; then
    printf 'missing contract tool: %s\n' "$contract_tool" >&2
    exit 1
fi

snapshot_root="$repo_root/models/DeepSeek-V4-Flash-0731-INT4-W4A16/models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/snapshots"
snapshot_revision=unknown
if [[ -d "$snapshot_root" ]]; then
    snapshot_revision=$(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | head -n 1)
fi

external_reference_revision=unavailable
reference_root=${AEON_VLLM_REFERENCE_ROOT:-"$repo_root/../aeon-references/vllm"}
if [[ -d "$reference_root/.git" ]]; then
    external_reference_revision=$(git -C "$reference_root" rev-parse HEAD)
fi

mkdir -p "$(dirname "$output_path")"
AEON_GIT_COMMIT=$(git -C "$repo_root" rev-parse HEAD) \
AEON_MODEL_CONFIG_SHA256=$(sha256sum "$model_dir/config.json" | awk '{print $1}') \
AEON_MANIFEST_SHA256=$(sha256sum "$model_dir/model_manifest.json" | awk '{print $1}') \
AEON_TOKENIZER_SHA256=$(sha256sum "$model_dir/tokenizer.aeon" | awk '{print $1}') \
AEON_SOURCE_SNAPSHOT_REVISION="$snapshot_revision" \
AEON_EXTERNAL_REFERENCE_REVISION="$external_reference_revision" \
"$contract_tool" "$model_dir" --output "$output_path"

printf 'wrote Stage 0 contract metadata: %s\n' "$output_path"