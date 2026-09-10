#!/bin/sh
set -eu
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1788912000
export FORCE_SOURCE_DATE=1
mkdir -p build
case "${1:-}" in
  ""|--submission-only|--companion-only) ;;
  *) printf '%s\n' 'Usage: sh build.sh [--submission-only|--companion-only]' >&2; exit 2 ;;
esac
if [ "${1:-}" != "--companion-only" ]; then
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_submission.tex > build/ICAART2027_submission-pass1.log
BIBINPUTS=.: BSTINPUTS=.: bibtex build/ICAART2027_submission > build/bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_submission.tex > build/ICAART2027_submission-pass2.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_submission.tex > build/ICAART2027_submission-pass3.log
cp build/ICAART2027_submission.pdf .
fi
if [ "${1:-}" != "--submission-only" ]; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_companion.tex > build/ICAART2027_companion-pass1.log
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_companion.tex > build/ICAART2027_companion-pass2.log
  cp build/ICAART2027_companion.pdf .
fi
