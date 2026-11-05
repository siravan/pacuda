#!/bin/sh
# SPDX-License-Identifier: 0BSD
# The runner's synthetic-plan test (was symjit-utils `recurrence_runner_variants_agree`):
# f64 sorted == unsorted bit for bit; df64 within 1e-9, mixed/mixed64 within 1e-4 of the
# largest value; each kernel's device reduction == host reduction (checked inside rectest).
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
dir=${TMPDIR:-/tmp}/pacuda-rectest-$$
mkdir -p "$dir"
trap 'rm -rf "$dir"' EXIT
"$here/runner/build.sh" "$dir" >/dev/null
cc -O1 -ffp-contract=off -I"$here/runner" -o "$dir/rectest" "$here/runner/rectest.c" -ldl -lm
run() { (cd "$dir" && ./rectest "$dir/librecurrence.so" "$@"); }
run f64 > "$dir/f64" || { [ $? = 77 ] && echo "skipped: no GPU" && exit 0; exit 1; }
run f64 nosort > "$dir/f64u"
run df64 > "$dir/df64"
run > "$dir/default"
run mixed > "$dir/mixed"
run mixed64 > "$dir/mixed64"
cmp -s "$dir/f64" "$dir/f64u" || { echo "f64 unsorted differs from f64" >&2; exit 1; }
python3 - "$dir" <<'EOF'
import sys
d = sys.argv[1]
read = lambda k: [float(v) for v in open(f"{d}/{k}").read().split()]
want = read("f64")
assert len(want) == 3 * 300 * 2 and any(want)
scale = max(abs(v) for v in want)
for k, tol in [("df64", 1e-9), ("default", 1e-9), ("mixed", 1e-4), ("mixed64", 1e-4)]:
    err = max(abs(g - w) for g, w in zip(read(k), want)) / scale
    assert err <= tol, (k, err)
    print(f"{k}: {err:.1e} (<= {tol})")
print("runner variants agree")
EOF
