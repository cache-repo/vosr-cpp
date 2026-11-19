#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$HERE/build" -j"$(nproc)"
echo "产物: $HERE/build/vosr-mnn-cpp"
