#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model_dir>" >&2
  exit 1
fi

model_dir="$1"

if [[ ! -d "$model_dir" ]]; then
  echo "Error: model_dir not found: $model_dir" >&2
  exit 1
fi

config_path="${model_dir}/config.yaml"
if [[ ! -f "$config_path" ]]; then
  echo "Error: config.yaml not found: $config_path" >&2
  exit 1
fi

base_name="$(awk -F': ' '/^basename:/{print $2; exit}' "$config_path")"
if [[ -z "$base_name" ]]; then
  echo "Error: basename not found in config.yaml" >&2
  exit 1
fi

num_chunks="$(awk -F': ' '/^num_chunks:/{print $2; exit}' "$config_path")"
if ! [[ "$num_chunks" =~ ^[0-9]+$ ]] || [[ "$num_chunks" -lt 1 ]]; then
  echo "Error: num_chunks not found or invalid in config.yaml" >&2
  exit 1
fi

for ((i=1; i<=num_chunks; i++)); do
  chunk_name="${base_name}_chunk${i}of${num_chunks}.mlpackage"
  chunk_path="${model_dir}/${chunk_name}"

  if [[ ! -d "$chunk_path" ]]; then
    echo "Warning: chunk not found, skip: $chunk_path" >&2
    continue
  fi

  xcrun coremlc compile "$chunk_path" "$model_dir"
  rm -rf "$chunk_path"
done
