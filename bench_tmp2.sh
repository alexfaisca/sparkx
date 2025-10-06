#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for ds in ./datasets/graphs/kmer_A2a.mtx; do
	[[ -f "$ds" ]] || continue
	for t in {0..2}; do
		echo ">> $(basename "$ds")  -t $t"
		GLIBC_TUNABLES=glibc.malloc.arena_max=1 cargo cache --tool=massif-pages --dataset="$ds" -t "$t"
	done
done
