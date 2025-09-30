#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for ds in ./datasets/graphs/synth*.mtx; do
	[[ -f "$ds" ]] || continue
	for t in {0..3}; do
		echo ">> $(basename "$ds")  -t $t"
		cargo cache --tool=massif --dataset="$ds" -t "$t"
	done
	for t in {3..10}; do
		echo ">> $(basename "$ds")  -t $t"
		cargo cache --tool=massif-pages --dataset="$ds" -t "$t"
	done
done
