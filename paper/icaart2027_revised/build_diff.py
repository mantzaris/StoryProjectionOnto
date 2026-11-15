"""Create a word-level LaTeX diff plus full-size before/after Figure 1 pages."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "icaart2027"
BUILD = HERE / "build"


def run(args, cwd=HERE, log="diff-command.log"):
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1794614400", "FORCE_SOURCE_DATE": "1",
           "BIBINPUTS": ".:", "BSTINPUTS": ".:"}
    result = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (BUILD / log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed; see {BUILD / log}\n{result.stdout[-3000:]}")


def flatten(folder, filename, bibliography):
    source = (folder / filename).read_text()
    source = source.replace(r"\input{generated/numbers}", (folder / "generated/numbers.tex").read_text())
    source = source.replace(r"\input{figure_blocks}", "")
    definitions = (folder / "figure_blocks.tex").read_text()
    for name, figure in re.findall(r"\\newcommand\{\\(\w+)\}\{\s*(\\begin\{figure\*\}.*?\\end\{figure\*\})\s*\}", definitions, re.S):
        source = source.replace("\\" + name, figure)
    source = re.sub(r"\\input\{(tables/[^}]+)\}", lambda m: (folder / (m[1] + ".tex")).read_text(), source)
    source = source.replace(r"\bibliographystyle{apalike}", "")
    source = source.replace(r"\bibliography{references}", bibliography.read_text())
    # ulem cannot line-break a long question hidden inside one macro argument.
    # Expand only question macros in the body; the abstract's numerical macros stay unchanged.
    questions = dict(re.findall(r"\\newcommand\{\\(\w+Question)\}\{([^\n]+)\}", (folder / "generated/numbers.tex").read_text()))
    preamble, body = source.split(r"\begin{document}", 1)
    for name, question in questions.items():
        body = re.sub(r"\\" + name + r"\b", lambda _: question, body)
    source = preamble + r"\begin{document}" + body
    # Expose heading text to latexdiff: otherwise it comments out a deleted
    # \uppercase command and leaves only an empty, numbered section heading.
    source = re.sub(r"\\section\{\\uppercase\{([^{}]+)\}\}",
                    lambda match: r"\section{" + match[1].upper() + "}", source)
    assert r"\input{" not in source and r"\OrchardFigure" not in source
    return source


def figure_proofs():
    from pdfrw import PdfReader, PdfWriter, PageMerge
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/storyprojection-paper-mpl")
    from matplotlib.font_manager import FontProperties, findfont

    page_width, page_height = A4
    frame = BUILD / "figure-proof-frames.pdf"
    for name, weight in [("ProofTimes", "normal"), ("ProofTimesBold", "bold")]:
        pdfmetrics.registerFont(TTFont(name, findfont(FontProperties(family="Times New Roman", weight=weight), fallback_to_default=False)))
    c = canvas.Canvas(str(frame), pagesize=A4, initialFontName="ProofTimes", initialFontSize=10)
    for label in ("Before: submitted Figure 1", "After: revised Figure 1"):
        c.setFont("ProofTimesBold", 15)
        c.drawString(40, page_height - 40, label)
        c.setFont("ProofTimes", 10)
        c.drawString(40, page_height - 58, "The revision expands the input text. The graph records and scores are unchanged.")
        c.showPage()
    c.save()
    pages = PdfReader(str(frame)).pages
    for page, folder in zip(pages, (OLD, HERE), strict=True):
        figure = PdfReader(str(folder / "figures/orchard_network.pdf")).pages[0]
        merge = PageMerge(page)
        item = merge.add(figure)[-1]
        item.scale((158.0134 / 25.4 * 72) / float(item.w))
        item.x = (page_width - float(item.w)) / 2
        item.y = page_height - 80 - float(item.h)
        assert item.y >= 35, "Figure comparison must fit the page"
        merge.render()
    return pages


def main():
    from pdfrw import PdfReader, PdfWriter

    BUILD.mkdir(exist_ok=True)
    baseline = BUILD / "baseline"
    baseline.mkdir(exist_ok=True)
    for rel in ["ICAART2027_submission.tex", "references.bib", "article.cls", "SCITEPRESS.sty",
                "apalike.sty", "apalike.bst", "figure_blocks.tex", "generated/numbers.tex",
                "tables/compact.tex", "tables/prose.tex", "figures/orchard_network.pdf",
                "figures/fable.pdf", "figures/alice.pdf"]:
        (baseline / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD / rel, baseline / rel)
    run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "ICAART2027_submission.tex"], baseline, "baseline-pass1.log")
    run(["bibtex", "ICAART2027_submission"], baseline, "baseline-bibtex.log")
    old_flat = BUILD / "submitted_flat.tex"
    new_flat = BUILD / "revised_flat.tex"
    old_flat.write_text(flatten(OLD, "ICAART2027_submission.tex", baseline / "ICAART2027_submission.bbl"))
    new_flat.write_text(flatten(HERE, "ICAART2027_revised.tex", BUILD / "ICAART2027_revised.bbl"))
    result = subprocess.run(["latexdiff", "--encoding=utf8", "--type=UNDERLINE", "--graphics-markup=none",
                             "--disable-citation-markup", str(old_flat), str(new_flat)],
                            capture_output=True, text=True, check=True)
    (BUILD / "latexdiff.log").write_text(result.stderr)
    source = result.stdout
    source = source.replace(r"\includegraphics[width=\textwidth]", r"\includegraphics[width=\textwidth,height=.55\textheight,keepaspectratio]")
    marker = r"\setcounter{footnote}{0}\vfill"
    assert marker in source
    source = source.replace(marker, marker + "\n" + r"\noindent{\small\textcolor{blue}{Blue underlined text is added.} \textcolor{red}{Red struck text is removed.} Title and abstract are unchanged. Figures are reduced here to fit deleted captions; full-size Figure 1 before/after pages follow.}\par\medskip", 1)
    (HERE / "ICAART2027_diff.tex").write_text(source)
    for i in range(2):
        run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "ICAART2027_diff.tex"], log=f"diff-pass{i+1}.log")
    writer = PdfWriter()
    writer.addpages(PdfReader(str(BUILD / "ICAART2027_diff.pdf")).pages)
    writer.addpages(figure_proofs())
    writer.write(str(HERE / "ICAART2027_diff.pdf"))
    print("Wrote ICAART2027_diff.pdf: text/caption/table/reference changes, then two Figure 1 comparison pages.")


if __name__ == "__main__":
    main()
