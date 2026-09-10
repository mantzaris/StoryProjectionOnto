"""Focused local publishing checks and clean compilation package; no experimental runs."""
from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from build_assets import HERE, ROOT, read, digest, write_json, companion_values


def command(*args, cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)


def normalized(s):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def abstract_body(source):
    """Read the inline declaration before the template's title-layout command."""
    match = re.search(r'(?ms)^\\abstract\{(.*?)\}\s*\\onecolumn', source)
    assert match, 'Expected inline abstract immediately before the title layout'
    return match[1]


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


def compare_pdf_pages(before, after):
    """Compare layout text and every rendered pixel, without relying on PDF metadata."""
    old = command('pdftotext', '-layout', str(before), '-')
    new = command('pdftotext', '-layout', str(after), '-')
    assert old == new, 'Companion layout text changed'
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='icaart-page-comparison-') as folder:
        folder = Path(folder)
        for label, pdf in [('before', before), ('after', after)]:
            command('pdftoppm', '-r', '150', '-png', str(pdf), str(folder/label))
        a, b = sorted(folder.glob('before-*.png')), sorted(folder.glob('after-*.png'))
        assert len(a) == len(b) and a
        pages = []
        for left, right in zip(a, b):
            with Image.open(left) as x, Image.open(right) as y:
                assert x.size == y.size and x.convert('RGB').tobytes() == y.convert('RGB').tobytes(), right.name
                pages.append(dict(size_px=list(x.size), rgb_sha256=hashlib.sha256(x.convert('RGB').tobytes()).hexdigest()))
    return dict(pages=len(pages), text_identical=True, pixels_identical=True, dpi=150,
                layout_text_sha256=hashlib.sha256(old.encode()).hexdigest(), rendered_pages=pages)


def verify_companion(compare_with=None):
    """Compile the authoritative single source with only its class and figure images."""
    filename = 'ICAART2027_companion.tex'
    source = (HERE/filename).read_text()
    active = re.sub(r'(?m)(?<!\\)%.*$', '', source)
    forbidden = r'\\(?:input|include|includeonly|import|subimport|inputfrom|subfile|subfileinclude|VerbatimInput|BVerbatimInput|LVerbatimInput|lstinputlisting|inputminted|bibliography|addbibresource)\b'
    assert not re.search(forbidden, active), 'Companion loads external document content'
    images = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', active)
    manifest = read(HERE/'manifest.json')
    assert images == [f'figures/{n}.pdf' for n in manifest['supplementary_figures']]
    for f in images:
        assert digest(HERE/f) == manifest['figures'][Path(f).stem]['output_sha256']['pdf']
    assert digest(HERE/'article.cls') == manifest['template_files']['article.cls'] == digest(HERE/'vendor/original/article.cls')
    for path, h in manifest['evidence_preservation']['complete_evidence_files'].items():
        assert digest(ROOT/path) == h, path
    datasets = {'compact':read(ROOT/'reports/tables/compact_story_v2_results.json'),
                'prose':read(ROOT/'reports/tables/real_text_proof_of_concept.json')}
    declarations, tail = companion_values(datasets)
    # Main-paper numerical macros remain a generated file. They are verification
    # inputs only here, never a companion compilation dependency or rewrite source.
    declarations += (HERE/'generated/numbers.tex').read_text().splitlines()
    for declaration in declarations:
        assert normalized(declaration) in normalized(source), declaration
    listing = re.search(r'\\begin\{Verbatim\}(?:\[[^\]]*\])?\n(.*?)\n\\end\{Verbatim\}', source, re.S)
    assert listing and listing[1] == tail, 'Inline raw-output excerpt differs from retained bytes'
    files = [filename, 'article.cls', *images]
    with tempfile.TemporaryDirectory(prefix='icaart-companion-source-') as folder:
        clean = Path(folder)
        for f in files:
            (clean/f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE/f, clean/f)
        assert sorted(str(p.relative_to(clean)) for p in clean.rglob('*') if p.is_file()) == sorted(files)
        env = {**os.environ, 'TEXINPUTS':'.:', 'BIBINPUTS':'.:',
               'SOURCE_DATE_EPOCH':'1788912000', 'FORCE_SOURCE_DATE':'1'}
        for _ in range(2):
            subprocess.run(['pdflatex','-no-shell-escape','-recorder','-interaction=nonstopmode',
                            '-halt-on-error',filename],cwd=clean,env=env,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log = (clean/'ICAART2027_companion.log').read_text()
        for failure in ['Overfull \\hbox', 'Overfull \\vbox', 'undefined', 'Label(s) may have changed']:
            assert failure not in log, failure
        for line in (clean/'ICAART2027_companion.fls').read_text().splitlines():
            if line.startswith('INPUT '):
                assert not (clean/line[6:]).resolve().is_relative_to(ROOT), line
        pdf = clean/'ICAART2027_companion.pdf'
        info, _ = pdf_check(pdf, main=False)
        assert 6 <= info['pages'] <= 8
        comparison = compare_pdf_pages(HERE/pdf.name, pdf)
    result = dict(single_editable_source=filename, source_sha256=digest(HERE/filename),
                  pdf_sha256=digest(HERE/'ICAART2027_companion.pdf'),
                  initial_isolated_files=files, no_external_content_commands=True,
                  no_repository_files_loaded=True, inline_data_and_listing='pass',
                  isolated_compilation='pass', pdf=info, isolated_comparison=comparison)
    if compare_with:
        result['prior_pdf_sha256'] = digest(compare_with)
        result['prior_pdf_comparison'] = compare_pdf_pages(Path(compare_with), HERE/'ICAART2027_companion.pdf')
    write_json(HERE/'companion_verification.json', result)
    print(f"Companion: {info['pages']} pages; single-source isolated compilation, data checks and all-page pixel comparison passed.")
    return result


def main():
    manifest=read(HERE/'manifest.json')
    for kind, filename in manifest['manuscript_roots'].items():
        assert filename == f'ICAART2027_{kind}.tex'
        assert (HERE/filename).is_file() and (HERE/filename).with_suffix('.pdf').is_file()
    assert digest(HERE/'references.bib')==manifest['conference_bibliography_sha256']
    assert digest(HERE/'REFERENCE_AUDIT.md')==manifest['reference_audit_sha256']
    for path,h in manifest['source_hashes'].items():assert digest(ROOT/path)==h,path
    for path,h in manifest['rendering_source_hashes'].items():assert digest(HERE/path)==h,path
    for path,h in manifest['template_files'].items():
        assert digest(HERE/path)==h==digest(HERE/'vendor/original'/path),path
    assert digest(HERE/'vendor/SCITEPRESS_Conference_Latex.zip')==manifest['template_archive_sha256']
    for dest,source in [('compact','compact_comparison'),('prose','published_comparison')]:
        assert read(HERE/f'tables/{dest}.json')==read(ROOT/f'paper/tables/{source}.json')
    # The entire retained output tables remain bound by their original report manifests.
    for stem in ['compact_story_v2','real_text_proof_of_concept']:
        original=read(ROOT/f'reports/tables/{stem}_manifest.json')
        for path,h in original.items():assert digest(ROOT/'reports'/path)==h,path
    _,datasets,panels,_=__import__('scripts.build_paper_figures',fromlist=['load_inputs']).load_inputs()
    from network_figures import load_panels
    panels.update(load_panels({'compact':datasets['synthetic'],'prose':datasets['real']}))
    for path,h in manifest['evidence_preservation']['complete_evidence_files'].items():
        assert digest(ROOT/path)==h,path
    for name,ds in [('compact',datasets['synthetic']),('prose',datasets['real'])]:
        stored=read(HERE/'data/retained_outputs.json')[name]
        for call in stored['calls']:
            original=next(c for c in ds['calls'] if c['case_id']==call['case_id'])
            assert all(original[k]==v for k,v in call.items()),call['case_id']
        assert stored['references']==ds['references']
    network_counts={}
    for name,f in manifest['figures'].items():
        for ext,h in f['output_sha256'].items():assert digest(HERE/f'figures/{name}.{ext}')==h
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
            if 'source_indices' in record:
                assert record['source_indices']==original['source_indices']
                assert record['source_call']==panels[record['panel']]['source_call']
                fact=record['fact']
                label=record['display_label']
                assert all(e in label for e in fact['evidence_ids'])
                if fact.get('valid_from') is not None or fact.get('valid_until') is not None:
                    assert f"[{fact.get('valid_from')},{fact.get('valid_until')})" in label
                if fact.get('holder') is not None:
                    assert f"{fact['holder']} / {fact['attitude']}" in label
        for panel in f.get('network_panels',[]):
            source=panels[panel['source_panel']]
            records=[r for r in f['records'] if r['panel']==panel['source_panel']]
            assert [r['index'] for r in records]==panel['indices']
            identities={r['fact'][k] for r in records for k in ('subject','object')}
            assert sorted(identities)==sorted(n['identity'] for n in panel['nodes'])
            assert len(identities)==panel['node_count']
            assert len(records)==panel['edge_count']==source['original_count']
            assert panel['omitted_record_count']==0
            # An index per retained record, never deduplication of parallel assertions.
            assert len(set(r['index'] for r in records))==len(records)
            adjacency={n:set() for n in identities}
            for r in records:
                a,b=r['fact']['subject'],r['fact']['object']
                adjacency[a].add(b);adjacency[b].add(a)
            remaining=set(identities); components=0
            while remaining:
                todo=[remaining.pop()];components+=1
                while todo:
                    adjacent=adjacency[todo.pop()] & remaining
                    remaining-=adjacent;todo.extend(adjacent)
            network_counts[name+'/'+panel['id']]=dict(nodes=len(identities),records=len(records),components=components)
            if panel['source_panel']=='orchard_A':
                assert (len(identities),len(records),components)==(6,5,1)
                pre=panels['orchard_pre']['records']
                assert all(r['fact']==pre[r['source_indices'][0]-1]['fact'] for r in records)
        svg=(HERE/f'figures/{name}.svg').read_text()
        assert '@font-face' in svg and '<text' in svg
        assert "local('Times New Roman')" in svg and 'data:font/' not in svg
    for filename in ['ICAART2027_submission.log','ICAART2027_companion.log']:
        log=(HERE/'build'/filename).read_text()
        assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log,filename
        assert 'undefined' not in log.lower(),filename
        assert 'Label(s) may have changed' not in log,filename
    report,text=pdf_check(HERE/'ICAART2027_submission.pdf',main=True)
    supp,_=pdf_check(HERE/'ICAART2027_companion.pdf',main=False)
    assert 6<=supp['pages']<=8,supp['pages']
    fig_chars=sum(len(re.sub(r'\s','',command('pdftotext',str(HERE/f'figures/{n}.pdf'),'-'))) for n in manifest['main_figures'])
    # All figures use searchable embedded text. Count again as a conservative upper estimate,
    # even though it is already in main PDF extraction; no graph labels are excluded.
    conservative=report['nonwhitespace_characters']+fig_chars
    assert 8000<=report['nonwhitespace_characters']<=conservative<40000
    abstract=abstract_body((HERE/'ICAART2027_submission.tex').read_text())
    for k,v in manifest['numerical_claims'].items():abstract=abstract.replace('\\'+k,v)
    abstract=abstract.replace('\\cite{openai2025}','(OpenAI, 2025)').replace('\\ ',' ')
    assert '\\' not in abstract,abstract
    aw=len(abstract.split())
    assert 70<=aw<=200,aw
    # Cite-to-entry correspondence; every entry must be used. No unverified self-citations.
    source='\n'.join((HERE/p).read_text() for p in ['ICAART2027_submission.tex','figure_blocks.tex'])
    cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',source) for k in group.split(',')}
    entries=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text()))
    assert cited==entries,(cited,entries)
    for number in ['0.625','0.308','454.82','79','249.45','205.37']:assert number in text,number
    # Honest anonymous package of only compilation inputs. Reproducible archive timestamps.
    files=['ICAART2027_submission.tex','figure_blocks.tex','references.bib','build.sh',
           'article.cls','SCITEPRESS.sty','apalike.sty','apalike.bst','generated/numbers.tex',
           'tables/compact.tex','tables/prose.tex']+[f'figures/{n}.pdf' for n in manifest['main_figures']]
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
        network_panels=network_counts,
        main_figures=len(manifest['main_figures']),main_result_tables=2,
        supplementary_figures=len(manifest['supplementary_figures']),
        historical_companion_pages=36,companion_pages_removed=36-supp['pages'],
        underlying_evidence_preserved='exact raw, parsed and reference comparison passed',
        figure_typeface='Times New Roman, embedded PDF subsets; editable SVG uses local faces',
        reference_audit_sha256=manifest['reference_audit_sha256'],
        source_zip=dict(sha256=digest(archive),files=files,isolated_compilation='pass',pdf_text_identical=True),
        no_new_inference=True,visual_inspection='See visual_inspection.md; generated after rendering, not asserted by automated checks.',
        companion_single_source=verify_companion(),
        outputs={str(p.relative_to(HERE)):digest(p) for p in [HERE/'ICAART2027_submission.tex',HERE/'ICAART2027_companion.tex',HERE/'references.bib',HERE/'figure_blocks.tex']})
    write_json(HERE/'verification.json',verification)
    print(json.dumps({k:verification[k] for k in ['abstract_words','conservative_double_counted_figure_bound']},indent=2))
    print('Submission:',report['pages'],'pages;',report['nonwhitespace_characters'],'non-whitespace characters. Companion:',supp['pages'],'pages. Clean source ZIP compiled; text identical.')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companion-only', action='store_true')
    parser.add_argument('--compare-with', type=Path, help='Prior companion PDF for organizational-change comparison')
    args = parser.parse_args()
    if args.compare_with and not args.companion_only:
        parser.error('--compare-with requires --companion-only')
    if args.companion_only:
        verify_companion(args.compare_with)
    else:
        main()
