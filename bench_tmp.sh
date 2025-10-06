#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for ds in ./datasets/graphs/kmer_V1r.mtx; do
	for t in {1..2}; do
		echo ">> $(basename "$ds")  -t $t"
		GLIBC_TUNABLES=glibc.malloc.arena_max=1 cargo cache --tool=massif --dataset="$ds" -t "$t"
	done
done
