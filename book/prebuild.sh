#!/usr/bin/env bash

set -euo pipefail

book="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$book")"

mdbook-langtabs install "$root"

fixup() {
    sed -i 's/#000000/var(--fg)/g;s/#000002/var(--icons)/g;s/#000005/var(--sidebar-active)/g;/^<svg/,$!d' "$@"
}

( cd "$book/math"
  python3 special/figures.py
  python3 integrate/figures.py )
fixup "$book"/math/special/*.svg "$book"/math/integrate/*/*.svg

while IFS= read -r -d '' script; do
    ( cd "$(dirname "$script")" && python3 "$(basename "$script")" )
done < <(find "$book/reproductions" -name '*.py' -print0)
while IFS= read -r -d '' svg; do
    fixup "$svg"
done < <(find "$book/reproductions" -name '*.svg' -print0)
