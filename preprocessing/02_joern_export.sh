#!/usr/bin/env bash
# Step 2: export every CPG (.bin) to GraphSON with Joern.
# For a function file <name>, Joern writes <JSON_OUT_DIR>/<name>/<name>/_global_.json
# (file-level graph, used in step 3) and <JSON_OUT_DIR>/<name>/<includes>/_global_.json (not used).
# Usage: ./02_joern_export.sh <JOERN_CLI_DIR> <CPG_DIR> <JSON_OUT_DIR>
set -uo pipefail
JOERN_DIR="$1"; CPG_DIR="$2"; OUT_DIR="$3"
mkdir -p "$OUT_DIR"
shopt -s nullglob
for bin_file in "$CPG_DIR"/*.bin; do
    out="$OUT_DIR/$(basename "$bin_file" .bin)"
    if "$JOERN_DIR/joern-export" "$bin_file" -o "$out" --repr cpg --format graphson; then
        echo "ok: $bin_file"
    else
        echo "$bin_file" >> "$OUT_DIR/failed_files.log"
    fi
done
echo "Step 2 done: $OUT_DIR"
