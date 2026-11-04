#!/bin/sh
# SPDX-License-Identifier: 0BSD
# Builds librecurrence.so from recurrence_runner.cu.
# usage: runner/build.sh [OUT_DIR] [SM...]   (SMs default to the local GPUs', e.g. 90 120)
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/../build}
[ $# -gt 0 ] && shift
sms="$*"
if [ -z "$sms" ]; then
    sms=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u)
fi
[ -n "$sms" ] || { echo "no GPU found: give the SMs (e.g. 90 120)" >&2; exit 1; }
gencode=""
for sm in $sms; do gencode="$gencode -gencode=arch=compute_$sm,code=sm_$sm"; done
mkdir -p "$out"
${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)} -O3 -fmad=false -Xcompiler -fPIC -shared -cudart static $gencode \
    -o "$out/librecurrence.so" "$here/recurrence_runner.cu"
echo "$out/librecurrence.so (sm $sms)"
