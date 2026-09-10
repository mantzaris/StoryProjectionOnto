"""Focused local publishing checks and clean compilation package; no experimental runs."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from build_assets import HERE, ROOT, read, digest, write_json


def command(*args, cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)


def normalized(s):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def pdf_check(path, *, main):
    info=command('pdfinfo',str(path))
    text=command('pdftotext','-enc','UTF-8',str(path),'-')
    assert re.search(r'^Author:\s*$',info,re.M),info
    assert '595.276 x 841.89' in info
    pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
    if main:assert pages<=8,pages
    bad=['mantzaris','StoryProjectionOnto','/home/resort','artifacts/restricted','ORCID','@gmail',
         'github.com/mantzaris','file://','root@','213.173.110.36']
    for needle in bad:assert needle.lower() not in (info+text).lower(),(path,needle)
    fonts=command('pdffonts',str(path))
    assert 'Type 3' not in fonts,fonts
    for line in fonts.splitlines()[2:]:
        if not line.strip():continue
        # emb/sub/uni columns are followed by object and generation ID.
        assert line.split()[-5]=='yes',line
    bbox=command('pdftotext','-bbox',str(path),'-')
    root=ET.fromstring(bbox)
    ns={'h':'http://www.w3.org/1999/xhtml'}
    page_bounds=[]
    table_captions=[]
    for i,p in enumerate(root.findall('.//h:page',ns),1):
        words=p.findall('.//h:word',ns)
        if not main:
            # The single-column companion has page numbers below its body margin.
            # Exclude only that exact page-number footer, never body text.
            words=[w for w in words if not (w.text==str(i) and float(w.attrib['yMin'])>780)]
        bb=[min(float(w.attrib['xMin']) for w in words),min(float(w.attrib['yMin']) for w in words),
            max(float(w.attrib['xMax']) for w in words),max(float(w.attrib['yMax']) for w in words)]
        # Small font bearings can extend <1 pt beyond the nominal left margin.
        if main:assert bb[0]>=72 and bb[2]<=523 and bb[1]>=90 and bb[3]<=727,(i,bb)
        else:assert bb[0]>=64 and bb[2]<=531 and bb[1]>=45 and bb[3]<=798,(i,bb)
        page_bounds.append(bb)
        if main:
            for w in words:
                # Concrete regression: the top table caption previously began at
                # y=91.59 pt, above the style's approximately 94.68 pt text margin.
                if w.text=="Table" and float(w.attrib['yMin'])<105:
                    y=float(w.attrib['yMin'])
                    assert y>=94.67,(i,"table caption above nominal margin",y)
                    table_captions.append(dict(page=i,y_min_pdf_points=y))
    return dict(pages=pages,nonwhitespace_characters=len(re.sub(r'\s','',text)),
                page_bounds_pdf_points=page_bounds,all_fonts_embedded=True,no_type3_fonts=True,
                author_metadata_empty=True,identifying_string_scan='pass',
                top_table_captions=table_captions,sha256=digest(path)),text


def main():
    manifest=read(HERE/'manifest.json')
    assert digest(HERE/'references.bib')==manifest['conference_bibliography_sha256']
    assert digest(HERE/'REFERENCE_AUDIT.md')==manifest['reference_audit_sha256']
    for path,h in manifest['source_hashes'].items():assert digest(ROOT/path)==h,path
    for path,h in manifest['template_files'].items():
        assert digest(HERE/path)==h==digest(HERE/'vendor/original'/path),path
    assert digest(HERE/'vendor/SCITEPRESS_Conference_Latex.zip')==manifest['template_archive_sha256']
    for dest,source in [('compact','compact_comparison'),('prose','published_comparison')]:
        assert read(HERE/f'tables/{dest}.json')==read(ROOT/f'paper/tables/{source}.json')
    # The entire retained output tables remain bound by their original report manifests.
    for stem in ['compact_story_v2','real_text_proof_of_concept']:
        original=read(ROOT/f'reports/tables/{stem}_manifest.json')
        for path,h in original.items():assert digest(ROOT/'reports'/path)==h,path
    _,_,panels,_=__import__('scripts.build_paper_figures',fromlist=['load_inputs']).load_inputs()
    for name,f in manifest['figures'].items():
        assert abs(f['width_mm']-158.0134)<.001
        assert f['minimum_font_pt']>=8.5
        assert f['font_family']=='Times New Roman'
        font_listing=command('pdffonts',str(HERE/f'figures/{name}.pdf'))
        assert 'TimesNewRomanPSMT' in font_listing and 'TimesNewRomanPS-BoldMT' in font_listing
        assert 'DejaVu' not in font_listing
        text=command('pdftotext',str(HERE/f'figures/{name}.pdf'),'-')
        assert len(normalized(text))>300
        for record in f['records']:
            original=next(r for r in panels[record['panel']]['records'] if r['index']==record['index'])
            assert record['fact']==original['fact'] and record['status']==original['status']
        svg=(HERE/f'figures/{name}.svg').read_text()
        assert '@font-face' in svg and '<text' in svg
        assert "local('Times New Roman')" in svg and 'data:font/' not in svg
    for filename in ['main.log','companion.log']:
        log=(HERE/'build'/filename).read_text()
        assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log,filename
        assert 'undefined' not in log.lower(),filename
        assert 'Label(s) may have changed' not in log,filename
    report,text=pdf_check(HERE/'ICAART2027_submission.pdf',main=True)
    supp,_=pdf_check(HERE/'ICAART2027_companion.pdf',main=False)
    fig_chars=sum(len(re.sub(r'\s','',command('pdftotext',str(HERE/f'figures/{n}.pdf'),'-'))) for n in manifest['figures'])
    # All figures use searchable embedded text. Count again as a conservative upper estimate,
    # even though it is already in main PDF extraction; no graph labels are excluded.
    conservative=report['nonwhitespace_characters']+fig_chars
    assert 8000<=report['nonwhitespace_characters']<=conservative<40000
    abstract=(HERE/'abstract.tex').read_text().strip().removeprefix('\\abstract{')[:-1]
    for k,v in manifest['numerical_claims'].items():abstract=abstract.replace('\\'+k,v)
    abstract=abstract.replace('\\cite{openai2025}','(OpenAI, 2025)').replace('\\ ',' ')
    assert '\\' not in abstract,abstract
    aw=len(abstract.split())
    assert 70<=aw<=200,aw
    # Cite-to-entry correspondence; every entry must be used. No unverified self-citations.
    source='\n'.join((HERE/p).read_text() for p in ['main.tex','abstract.tex','figure_blocks.tex'])
    cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',source) for k in group.split(',')}
    entries=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text()))
    assert cited==entries,(cited,entries)
    for number in ['0.625','0.308','454.82','79','249.45','205.37']:assert number in text,number
    # Honest anonymous package of only compilation inputs. Reproducible archive timestamps.
    files=['main.tex','abstract.tex','figure_blocks.tex','references.bib','build.sh',
           'article.cls','SCITEPRESS.sty','apalike.sty','apalike.bst','generated/numbers.tex',
           'tables/compact.tex','tables/prose.tex']+[f'figures/{n}.pdf' for n in manifest['figures']]
    archive=HERE/'ICAART2027_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data=(HERE/f).read_bytes()
            for needle in [b'mantzaris',b'StoryProjectionOnto',b'/home/resort',b'author_actions',b'artifacts/restricted']:
                assert needle.lower() not in data.lower(),f
            zi=zipfile.ZipInfo(f,(2026,9,9,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
            zi.external_attr=0o644<<16;z.writestr(zi,data)
    clean=Path(tempfile.mkdtemp(prefix='icaart-source-check-'))
    with zipfile.ZipFile(archive) as z:z.extractall(clean)
    command('sh','build.sh','--submission-only',cwd=clean)
    clean_text=command('pdftotext',str(clean/'ICAART2027_submission.pdf'),'-')
    assert clean_text==text,'Isolated source ZIP PDF text mismatch'
    # A non-identifying submission metadata sheet; author fields remain local and pending.
    title='Inspectable Narrative Graphs: Evidence-Linked Extraction and Contextual Selection'
    (HERE/'submission_metadata.md').write_text('# Submission metadata (local draft)\n\n'
        f'Title: **{title}**\n\nCategory: **Position Paper**, empirical work in progress.\n\n'
        'Area: **2 — Artificial Intelligence**. Primary topic: **Natural Language Processing**. '
        'Additional topics, if multiple selections are permitted: Knowledge Representation and Reasoning; Large Language Models (LLMs); Visualization.\n\n'
        'Keywords: Natural Language Processing, Knowledge Representation and Reasoning, Large Language Models, Visualization.\n\n'
        'Deadline verified 9 September 2026: **22 October 2026, Anywhere on Earth (UTC−12)**. '
        'See [official dates](https://icaart.scitevents.org/ImportantDates.aspx).\n\n'
        f'## Abstract ({aw} whitespace-delimited words, citation included)\n\n{abstract}\n\n'
        '## Outstanding fields\n\nFull author names, affiliation, email, optional ORCID, contributions, funding, conflicts and presentation arrangements require author confirmation. '
        'They are intentionally absent from the review PDF. See author_actions.md for public-version chronology, semantic decisions and eligibility questions. '
        'No review decisions or submission identifiers have been invented.\n')
    verification=dict(submission=report,companion=supp,abstract_words=aw,
        character_count_method='pdftotext UTF-8, remove every Unicode whitespace character; includes title, abstract, captions, tables, searchable vector graph text, references and appendix. PDF discretionary hyphens retained. Counts are local, not a claimed PRIMORIS count.',
        figure_text_characters_already_included=fig_chars,conservative_double_counted_figure_bound=conservative,
        scheduled_character_range=[8000,40000],ordinary_page_limit=8,
        template_files_unmodified=True,retained_source_hashes='pass',canonical_tables='unchanged',
        record_and_assessment_correspondence='pass',references_cited_and_defined=sorted(cited),
        figure_typeface='Times New Roman, embedded PDF subsets; editable SVG uses local faces',
        reference_audit_sha256=manifest['reference_audit_sha256'],
        source_zip=dict(sha256=digest(archive),files=files,isolated_compilation='pass',pdf_text_identical=True),
        no_new_inference=True,visual_inspection='See visual_inspection.md; generated after rendering, not asserted by automated checks.',
        outputs={str(p.relative_to(HERE)):digest(p) for p in [HERE/'main.tex',HERE/'abstract.tex',HERE/'references.bib',HERE/'figure_blocks.tex',HERE/'generated/companion_content.tex']})
    write_json(HERE/'verification.json',verification)
    print(json.dumps({k:verification[k] for k in ['abstract_words','conservative_double_counted_figure_bound']},indent=2))
    print('Submission:',report['pages'],'pages;',report['nonwhitespace_characters'],'non-whitespace characters. Companion:',supp['pages'],'pages. Clean source ZIP compiled; text identical.')


if __name__=='__main__':main()
