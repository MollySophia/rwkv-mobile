#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <model_dir> <num_chunks>" >&2
  exit 1
fi

model_dir="$1"
num_chunks="$2"

if [[ ! -d "$model_dir" ]]; then
  echo "Error: model_dir not found: $model_dir" >&2
  exit 1
fi

if ! [[ "$num_chunks" =~ ^[0-9]+$ ]] || [[ "$num_chunks" -lt 1 ]]; then
  echo "Error: num_chunks must be a positive integer" >&2
  exit 1
fi

base_name="$(basename "$model_dir")"

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
