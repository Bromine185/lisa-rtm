#!/bin/bash
# Package this folder as the single supplemental file AAAI asks for (.zip, at most 20 MB).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../supplementary.zip}"
rm -f "$OUT"
zip -r -q -X "$OUT" . -x "build_zip.sh" -x "*/__pycache__/*" -x "*.pyc" -x ".DS_Store"
BYTES=$(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT")
echo "$OUT: $(( BYTES / 1024 )) KiB"
[ "$BYTES" -le 20000000 ] || { echo "over the 20 MB limit"; exit 1; }
