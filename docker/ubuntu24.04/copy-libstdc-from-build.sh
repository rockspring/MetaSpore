#!/bin/bash
set -euo pipefail
f=$(find /b/lib /b/usr/lib -maxdepth 2 -name 'libstdc++.so.6.0.*' -type f 2>/dev/null | head -1)
if [ -z "$f" ]; then
  echo "libstdc++.so.6 not found in build image" >&2
  exit 1
fi
rel=${f#/b}
mkdir -p "$(dirname "$rel")"
cp "$f" "$rel"
bn=$(basename "$f")
ln -svf "$bn" "$(dirname "$rel")/libstdc++.so.6"
