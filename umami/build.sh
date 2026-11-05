#!/bin/sh
# SPDX-License-Identifier: 0BSD
# Builds the pacuda UMAMI library of one mg7 subprocess.
# usage: umami/build.sh UMAMI_DIR RUNNER_DIR OUT.so [RPATH]
#   UMAMI_DIR: the subprocess's lc/API/umami (umami.h, umami_provider.h, provider_p0.h, umami.exports)
#   RUNNER_DIR: the directory of the output's librecurrence.so
#   RPATH: where the library finds librecurrence.so at run time (default RUNNER_DIR; '$ORIGIN'
#   when the runner is installed next to it)
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
umami=$1; runner=$2; out=$3; rpath=${4:-$2}
${CC:-cc} -O2 -std=c11 -Wall -Wextra -pedantic -fPIC -shared -Wl,-z,defs \
    -Wl,--version-script="$umami/umami.exports" -I"$umami" -I"$here/runner" \
    -o "$out" "$here/umami/umami_pacuda.c" -L"$runner" -lrecurrence -Wl,-rpath,"$rpath" -lm
