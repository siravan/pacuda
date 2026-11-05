#!/bin/sh
# SPDX-License-Identifier: 0BSD
# The patched RustiCol pacuda-convert builds on: pyAmpliCol (the pyamplicol/ directory of
# MadGraph7 at the commit in UPSTREAM) with the patches in patches/ (the pacman runner hook and its
# record mode, rusticol-core only), in build/rusticol of this repository.
# usage: rusticol/prepare.sh [MADGRAPH7_REPO]
#   MADGRAPH7_REPO: a local clone that has the UPSTREAM commit (else it is fetched from
#   github.com/MadGraphTeam/MadGraph7)
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
base=$(cat "$here/UPSTREAM")
out=$root/build/rusticol
stamp=$out/PREPARED
want="$base $(cat "$here"/patches/*.patch | sha256sum | cut -c1-64)"
if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
    echo "$out: up to date"
    exit 0
fi
rm -rf "$out"
mkdir -p "$out"
if [ -n "$1" ]; then
    git -C "$1" archive "$base" pyamplicol | tar -x -C "$out"
else
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git -C "$tmp" init -q
    git -C "$tmp" fetch -q --depth 1 --filter=blob:none https://github.com/MadGraphTeam/MadGraph7.git "$base"
    git -C "$tmp" archive "$base" pyamplicol | tar -x -C "$out"
fi
for p in "$here"/patches/*.patch; do
    # (patch, not git apply: inside this repository git apply would skip the paths outside it)
    patch -s -p1 -N --no-backup-if-mismatch -d "$out" < "$p" || { echo "prepare: $(basename "$p") does not apply" >&2; exit 1; }
done
echo "$want" > "$stamp"
echo "$out: pyamplicol at $base with $(ls "$here"/patches/*.patch | wc -l) patches"
