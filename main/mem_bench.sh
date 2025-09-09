#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for ds in ./datasets/graphs/*.{lz4,mtx}; do
	[[ -f "$ds" ]] || continue
	for t in {0..10}; do
		echo ">> $(basename "$ds")  -t $t"
		cargo cache --tool=massif-pages --dataset="$ds" -t "$t"
	done
done
