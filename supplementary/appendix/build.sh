#!/bin/bash
# Rebuild the tables from results.json, then the appendix PDF (needs a LaTeX engine: tectonic, or latexmk/pdflatex).
set -euo pipefail
cd "$(dirname "$0")"
python3 ../results/make_tables.py
if command -v tectonic >/dev/null; then tectonic -X compile technical_appendix.tex
else latexmk -pdf -interaction=nonstopmode technical_appendix.tex; fi
cp technical_appendix.pdf ../technical_appendix.pdf
