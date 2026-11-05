#!/bin/sh
# SPDX-License-Identifier: 0BSD
# Builds pacuda-convert: prepares the patched RustiCol (rusticol/prepare.sh, with its arguments),
# then cargo build --release (convert/target/release/pacuda-convert).
set -e
here=$(cd "$(dirname "$0")" && pwd)
"$here/../rusticol/prepare.sh" "$@"
cd "$here"
cargo build --release
