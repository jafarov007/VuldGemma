#!/usr/bin/env bash
# Step 1: build a Code Property Graph (CPG) for every C/C++ function file with Joern.
# Usage: ./01_joern_parse.sh <JOERN_CLI_DIR> <SOURCE_DIR> <CPG_OUT_DIR>
# Tested with Joern v4.0.227.
set -euo pipefail
JOERN_DIR="$1"; SRC_DIR="$2"; OUT_DIR="$3"
mkdir -p "$OUT_DIR"
shopt -s nullglob
for file in "$SRC_DIR"/*.c "$SRC_DIR"/*.cpp; do
    out="$OUT_DIR/$(basename "$file").bin"
    if [ -f "$out" ]; then
        echo "skip (exists): $out"; continue
    fi
    "$JOERN_DIR/joern-parse" "$file" --output "$out"
done
echo "Step 1 done: $OUT_DIR"
