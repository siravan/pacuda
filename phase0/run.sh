#!/bin/sh
# Phase 0: record the runner calls of the LC libraries of the outputs in phase0/outputs under
# varied inputs (momenta, batch size, alpha_s, helicity mode), plus a CPU reference run of each.
# usage: phase0/run.sh [OUTPUT...]   (default: every output); results in phase0/results/
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$here/..
res=$here/results
mkdir -p "$res"
[ -f "$root/build/librecurrence.so" ] || "$root/runner/build.sh" "$root/build"
cc -O2 -shared -fPIC -o "$root/build/librecord.so" "$here/record.c" -ldl -lpthread
outputs=${*:-$(ls "$here/outputs")}
export PACMAN_RECURRENCE_KERNEL=f64
for o in $outputs; do
    dir=$here/outputs/$o
    rm -f "$res/$o"-*
    # name seed points alpha_s mode
    while read -r name seed k a mode; do
        PACUDA_TRACE=$res/$o-$name.jsonl PACUDA_REAL_RUNNER=$root/build/librecurrence.so \
        PACMAN_RECURRENCE_RUNNER=$root/build/librecord.so \
            python3 "$here/drive.py" "$dir" "$res/$o-$name.npz" "$seed" "$k" "$a" "$mode" 2>>"$res/$o.log"
    done <<EOF
base 1 1000 0.118 sample
momenta 2 1000 0.118 sample
batch 1 300 0.118 sample
alphas 1 1000 0.13 sample
sum 1 1000 0.118 sum
EOF
    python3 "$here/drive.py" "$dir" "$res/$o-cpu.npz" 1 1000 0.118 sample 2>>"$res/$o.log"
    python3 "$here/drive.py" "$dir" "$res/$o-cpusum.npz" 1 1000 0.118 sum 2>>"$res/$o.log"
done
