#!/usr/bin/env sh

# Usage: flamegraph.sh [FlameGraph path] [svg opener]

mkdir -p flamegraphs

perf record -F 99 -g -o ./flamegraphs/perf.data -- ./basis-choice
perf script -i ./flamegraphs/perf.data > ./flamegraphs/out.perf
"$1"/stackcollapse-perf.pl ./flamegraphs/out.perf > ./flamegraphs/out.folded
"$1"/flamegraph.pl ./flamegraphs/out.folded > ./flamegraphs/out.svg
"$2" ./flamegraphs/out.svg
