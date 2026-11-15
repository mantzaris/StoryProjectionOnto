#!/bin/sh
set -eu
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1794614400
export FORCE_SOURCE_DATE=1
mkdir -p build
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_revised_v2.tex > build/revised-pass1.log
BIBINPUTS=.: BSTINPUTS=.: bibtex build/ICAART2027_revised_v2 > build/bibtex.log
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_revised_v2.tex > build/revised-pass2.log
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build ICAART2027_revised_v2.tex > build/revised-pass3.log
cp build/ICAART2027_revised_v2.pdf .
printf 'Revised manuscript: %s/ICAART2027_revised_v2.pdf\n' "$PWD"
