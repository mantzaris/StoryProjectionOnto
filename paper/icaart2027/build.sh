#!/bin/sh
set -eu
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1788912000
export FORCE_SOURCE_DATE=1
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/main-pass1.log
BIBINPUTS=.: BSTINPUTS=.: bibtex build/main > build/bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/main-pass2.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/main-pass3.log
cp build/main.pdf ICAART2027_submission.pdf
if [ "${1:-}" != "--submission-only" ]; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build companion.tex > build/companion-pass1.log
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build companion.tex > build/companion-pass2.log
  cp build/companion.pdf ICAART2027_companion.pdf
fi
