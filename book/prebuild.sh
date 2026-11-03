#!/usr/bin/env bash

set -euo pipefail

book="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$book")"

fixup() {
    sed -i 's/#000000/var(--fg)/g;s/#000002/var(--icons)/g;s/#000005/var(--sidebar-active)/g;/^<svg/,$!d' "$@"
}

( cd "$book/math"
  python3 special/figures.py
  python3 integrate/figures.py )
fixup "$book"/math/special/*.svg "$book"/math/integrate/*/*.svg
