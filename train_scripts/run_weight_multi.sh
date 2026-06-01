#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_weight_multi.sh [start_from_bottom]
# start_from_bottom is the number of runs to skip from the bottom of the list.
start_from_bottom="${1:-0}"

runs=(
	"multi55 0.5 0.5"
	"multi01 0.0 1.0"
	"multi28 0.2 0.8"
	"multi46 0.4 0.6"
	"multi64 0.6 0.4"
	"multi82 0.8 0.2"
	"multi10 1.0 0.0"
	"multi19 0.1 0.9"
	"multi37 0.3 0.7"
	"multi73 0.7 0.3"
	"multi91 0.9 0.1"
)

total_runs=${#runs[@]}

if [[ "$start_from_bottom" =~ ^[0-9]+$ ]] && (( start_from_bottom >= 0 && start_from_bottom < total_runs )); then
	for ((index=total_runs - 1 - start_from_bottom; index >= 0; index--)); do
		read -r run_name cost_weight delay_weight <<< "${runs[index]}"
		python run.py --name "$run_name" --cost_weight "$cost_weight" --delay_weight "$delay_weight" --cuda
	done
else
	echo "Usage: $0 [start_from_bottom]"
	echo "  start_from_bottom must be an integer between 0 and $((total_runs - 1))"
	exit 1
fi
