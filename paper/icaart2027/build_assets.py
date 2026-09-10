"""CPU-only conference publishing; no inference, scoring changes or service imports.

Run from any directory: python paper/icaart2027/build_assets.py
Only this package's generated artifacts are written. The prior publications stay intact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_paper_figures as graph

W = 6.221 * 25.4 / 25.4 * 72  # exactly the official style's 6.221 in, in PDF points
graph.W = W  # process-local width only; never regenerate or edit historical artwork
FONT = "Times New Roman"
# Require the installed face, rather than silently falling back to a sans serif.
FONT_FILES = {
    weight: Path(graph.findfont(graph.FontProperties(family=FONT, weight=weight),
                               fallback_to_default=False))
    for weight in ("normal", "bold")
}
graph.matplotlib.rcParams["font.family"] = FONT


@lru_cache(maxsize=20000)
def conference_text_width(text, size=9, bold=False):
    return graph.TextToPath().get_text_width_height_descent(
        text, graph.FontProperties(family=FONT, size=size,
                                   weight="bold" if bold else "normal"), False)[0]


graph.width = conference_text_width  # wrapping and rendering use the same face


def annotate_svg(path, plate):
    """Preserve searchable labels and record links without redistributing font files.

    PDF embeds document font subsets. Editable SVG requires a licensed local Times
    New Roman installation. PNG is the font-independent raster counterpart.
    """
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.parse(path).getroot()
    defs = root.find(f"{{{ns}}}defs")
    ET.SubElement(defs, f"{{{ns}}}style").text = (
        "@font-face{font-family:'Times New Roman';font-weight:normal;"
        "src:local('Times New Roman'),local('TimesNewRomanPSMT');}"
        "@font-face{font-family:'Times New Roman';font-weight:bold;"
        "src:local('Times New Roman Bold'),local('TimesNewRomanPS-BoldMT');}"
    )
    ET.SubElement(root, f"{{{ns}}}metadata").text = (
        "Conference figure. Searchable Times New Roman text requires the local font. "
        "No font software is redistributed in this SVG. PDF embeds document subsets."
    )
    for item in plate.edges:
        for gid in item["artist_ids"]:
            group = next(el for el in root.iter() if el.get("id") == gid)
            group.set("data-record", item["id"])
            group.set("role", "button")
            group.set("tabindex", "0")
            ET.SubElement(group, f"{{{ns}}}title").text = (
                item["status"] + ": " + json.dumps(item["fact"], ensure_ascii=False)
                + ". Citation link only. " + item["note"]
            )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def tex(s):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
               '→': r'$\rightarrow$', '−': '-', '≤': r'$\leq$', '≥': r'$\geq$',
               '–': '--', '—': '---', '’': "'", '‘': '`', '“': '``', '”': "''"}
    return ''.join(mapping.get(c, c) for c in str(s))


def make_plate(name):
    p = graph.Plate(name, '', '', 400)
    p.ax.clear()
    p.ax.set(xlim=(0, W), ylim=(400, 0))
    p.ax.axis('off')
    p.text_items = []
    return p


def edge(p, panel, r, x, y, width, *, sw=63, ow=64, height=32):
    """Exact-string endpoint facets, including duplicate records; no graph repair."""
    f = r['fact']
    color, symbol, style = graph.STATUS[r['status']]
    cy = y + height / 2
    for cx, value, bw in [(x+sw/2, f['subject'], sw), (x+width-ow/2, f['object'], ow)]:
        lines = graph.wrap(value, bw-5, 8.5)
        bh = max(22, len(lines)*11+6)
        p.rect(cx-bw/2, cy-bh/2, bw, bh)
        p.text(cx, cy-len(lines)*11/2, '\n'.join(lines), size=8.5, align='center')
    left, right = x+sw+2, x+width-ow-2
    a = graph.FancyArrowPatch((left, cy+6), (right, cy+6), arrowstyle='-|>',
                             mutation_scale=9, lw=1, color=color, linestyle=style)
    p.ax.add_patch(a)
    label = symbol+' '+f['relation'].replace('_', ' ')
    t = p.text((left+right)/2, y, '\n'.join(graph.wrap(label, right-left+4, 8.5)),
               size=8.5, color=color, align='center')
    t.set_bbox(dict(facecolor='white', edgecolor='none', pad=.4))
    b = p.text((left+right)/2, y+height-7, ' '.join(f['evidence_ids']), size=8.5,
               color=color, align='center')
    graph.register_edge(p, panel, r, [a,t,b])


def save_plate(p, height, note):
    p.height = height
    p.fig.set_size_inches(W/72, height/72)
    p.ax.set_ylim(height, 0)
    p.fig.canvas.draw()
    renderer = p.fig.canvas.get_renderer()
    for t in p.text_items:
        box = t.get_window_extent(renderer)
        fb = p.fig.bbox
        assert box.x0 >= fb.x0-1 and box.x1 <= fb.x1+1, (p.name, t.get_text(), 'x overflow')
        assert box.y0 >= fb.y0-1 and box.y1 <= fb.y1+1, (p.name, t.get_text(), 'y overflow')
    out = HERE/'figures'/p.name
    p.fig.savefig(out.with_suffix('.pdf'), metadata={'Author':'', 'Creator':'Matplotlib',
                  'Title':p.name, 'CreationDate':None, 'ModDate':None})
    p.fig.savefig(out.with_suffix('.svg'), metadata={'Creator':'Matplotlib', 'Date':None})
    annotate_svg(out.with_suffix('.svg'),p)
    p.fig.savefig(out.with_suffix('.png'), dpi=400)
    record = dict(width_mm=W/72*25.4, height_mm=height/72*25.4, minimum_font_pt=8.5,
                  font_family=FONT, font_sha256={k:digest(v) for k,v in FONT_FILES.items()},
                  svg_font_delivery="licensed local face; no redistributed font software",
                  text=[t.get_text() for t in p.text_items], records=p.edges,
                  display_transformation=note)
    record['output_sha256'] = {ext:digest(out.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}
    if hasattr(p, 'network_panels'):
        record['network_panels'] = p.network_panels
    graph.plt.close(p.fig)
    return record


def figures(data, panels):
    meta = {}
    p=make_plate('fable')
    p.text(5, 3, 'The Lion and the Mouse · actions and rescue', size=10, bold=True)
    text='S2 … "If you would only spare my life, I would be sure to repay your kindness."'
    y=p.para(5, 22, text, W-10, size=8.5)
    y=p.para(5, y+4, 'S5 '+ ' '.join(data['prose']['stories']['fable']['evidence']['S5'].split()), W-10, size=8.5)+8
    p.text(5,y,'Request: physical actions, excluding dialogue and intentions',size=8.5,bold=True)
    y+=20
    half=(W-20)/2
    p.text(5,y,'A: text → extraction → selection',size=8.5,bold=True)
    p.text(half+15,y,'B: text + question → extraction',size=8.5,bold=True)
    y+=21
    aa=[panels['fable_A']['records'][i-1] for i in [2,3,4,7,8]]
    bb=panels['fable_B']['records']
    for a,b in zip(aa,bb):
        edge(p,panels['fable_A'],a,5,y,half,sw=53,ow=53,height=37)
        edge(p,panels['fable_B'],b,half+15,y,half,sw=53,ow=53,height=37)
        y+=45
    y=p.para(5,y,'+ supported meaning   ~ partial meaning. Citations are not proof of support.',W-10,size=8.5)+4
    y=p.para(5,y,'A: spare loses conditional scope. B: early capture/release missing. Source excerpted. Full fable supplied.',W-10,size=8.5)+8
    meta['fable']=save_plate(p,y,'Five of eight A facts, all five B facts. Full exact strings; underscore-to-space labels and whitespace reflow only. Omitted A indices 1,5,6. Bounds and attribution null except actual missing valid_until (key valid until). No node merging across conditions.')

    p=make_plate('harbor')
    p.text(5,3,'Harbor · full intervals, not clipped to the question',size=10,bold=True)
    y=22
    for eid in ['S5','S6','S7','S8']:
        y=p.para(5,y,eid+' '+data['compact']['stories']['harbor']['evidence'][eid],W-10,size=8.5)+3
    y=p.para(5,y+3,'Request: narrated locations overlapping [2,4), with full source intervals.',W-10,size=8.5,bold=True)+8
    p.text(5,y,'A and B: four identical location records, generated independently',size=8.5,bold=True)
    y+=23
    x0,x1=W-117,W-6
    p.ax.add_patch(graph.Rectangle((x0+(x1-x0)*2/6,y-8),(x1-x0)*2/6,211,fc='#E1E9F2',zorder=0))
    for day in range(7):
        x=x0+(x1-x0)*day/6
        p.text(x,y-13,str(day),size=8.5,align='center')
    for i,r in enumerate(panels['harbor_A_locations']['records']):
        edge(p,panels['harbor_A_locations'],r,5,y+7,W-139,sw=66,ow=72,height=32)
        rb=panels['harbor_B_locations']['records'][i+2]
        assert r['fact']==rb['fact']
        graph.register_edge(p,panels['harbor_B_locations'],rb,[])
        f=r['fact']; cy=y+26
        xa=x0+(x1-x0)*f['valid_from']/6;xb=x0+(x1-x0)*f['valid_until']/6
        p.ax.plot([xa,xb],[cy,cy],color='#14776E',lw=2)
        p.ax.plot([xa],[cy],'o',ms=4,color='#14776E')
        p.ax.plot([xb],[cy],'o',ms=4,mfc='white',mec='#14776E')
        p.text((xa+xb)/2,cy-15,f"[{f['valid_from']},{f['valid_until']})",size=8.5,align='center')
        y+=48
    y+=10
    p.text(5,y,'B also returns two irrelevant carrying records (both still scored)',size=8.5,bold=True)
    y+=23
    for r in panels['harbor_B_locations']['records'][:2]:
        edge(p,panels['harbor_B_locations'],r,5,y,W-100,sw=66,ow=100,height=32)
        f=r['fact'];p.text(W-83,y+14,f"[{f['valid_from']},{f['valid_until']})",size=8.5)
        y+=42
    y=p.para(5,y,'+ supported   I irrelevant; all holder/attitude values null. Shading = query window; filled start / open end = source validity.',W-10,size=8.5)+7
    meta['harbor']=save_plate(p,y,'All four A and six B facts. Four identical location records share labelled display rows only; both provenance records retained. Two B carrying extras shown. Source S5-S8 of fourteen; no bounds changed.')

    p=make_plate('alice')
    p.text(5,3,'Alice · inspecting an action and a thought',size=10,bold=True)
    excerpts=[('S1','once or twice she had peeped into the book her sister was reading'),
              ('S1','"and what is the use of a book," thought Alice "without pictures or conversations?"'),
              ('S2','whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies')]
    y=22
    for eid,t in excerpts:
        # Exact words are verified against whitespace-normalized supplied evidence.
        ev=' '.join(data['prose']['stories']['alice']['evidence'][eid].split())
        assert t.replace('"','').replace(',','') in ev.replace('“','').replace('”','').replace('"','').replace(',',''),t
        y=p.para(5,y,eid+' … '+t+' …',W-10,size=8.5)+3
    y+=7;p.text(5,y,'B action request: sitting, reading/inspection and running',size=8.5,bold=True);y+=21
    for ix in [3,4]:
        edge(p,panels['alice_B_actions'],panels['alice_B_actions']['records'][ix-1],5,y,W-10,sw=80,ow=126,height=34);y+=42
    p.text(5,y,'x The sister reads. Alice peeps. Same source, different participant.',size=8.5,color='#B33C3D');y+=23
    p.text(5,y,'B claims request: absent book content and Alice’s thoughts',size=8.5,bold=True);y+=23
    for ix in [2,3]:
        edge(p,panels['alice_B_claims'],panels['alice_B_claims']['records'][ix-1],5,y,W-10,sw=85,ow=215,height=56);y+=65
    y=p.para(5,y,'Final edge: holder = Alice, attitude = null. Other qualifiers null. Thought remains a sentence. Daisy-chain judgment unresolved.',W-10,size=8.5)+5
    y=p.para(5,y,'+ supported meaning   x unsupported   ? unresolved. Source excerpted. Full passage supplied to each independent request.',W-10,size=8.5)+7
    meta['alice']=save_plate(p,y,'Action indices 3,4 of five; claim indices 2,3 of three. Exact endpoint strings retained, including sentence-valued object. Predicate underscores printed as spaces. No reconstructed belief graph. Source excerpts marked.')
    return meta


def numbers_and_tables(compact, prose):
    registry=read(ROOT/'paper/tables/numerical_registry.json')
    cm=read(ROOT/'paper/tables/compact_comparison.json')
    pm=read(ROOT/'paper/tables/published_comparison.json')
    alloc=read(ROOT/'paper/tables/allocation.json')
    numbers={
        'Temperature':registry['temperature']['display'],'TopP':registry['top_p']['display'],
        'TopK':registry['top_k']['display'],'Seed':'1988649846','OutputTokens':'2,048','ContextTokens':'12,288',
        'StorySentences':str(len(compact['stories']['harbor']['evidence'])),
        'FableWords':str(prose['stories']['fable']['word_count']), 'AliceWords':str(prose['stories']['alice']['word_count']),
        'HolmesWords':str(prose['stories']['holmes']['word_count']),
        'CompactCalls':str(len(compact['calls'])),'ProseCalls':str(len(prose['calls'])),
        'ProseParsed':str(sum(c['parsed'] is not None for c in prose['calls'])),
        'IntervalsKept':registry['interval_retained']['display'],'IntervalsEligible':registry['interval_eligible']['display'],
        'OrchardLocationRecall':registry['orchard_locations_B_recall']['display'],
        'MainCalls':str(sum(a['calls'] for a in alloc[2:])),
        'MainAllocation':f"{sum(a['allocated_seconds'] for a in alloc[2:]):.2f}",
        'CompactAllocation':f"{alloc[2]['allocated_seconds']:.2f}",'ProseAllocation':f"{alloc[3]['allocated_seconds']:.2f}",
        'CompactRequest':f"{alloc[2]['request_seconds']:.2f}",'ProseRequest':f"{alloc[3]['request_seconds']:.2f}",
    }
    for prefix,story,method,task in [('FableA','fable','A','actions'),('FableB','fable','B','actions'),('AliceA','alice','A','claims'),('AliceB','alice','B','claims')]:
        r=next(r for r in pm if r['story']==story and r['approach'].startswith(method) and r['task']==task)
        numbers[prefix+'Fone']=f"{r['strict_qualified_f1']:.3f}"
        numbers[prefix+'Coverage']=f"{r['semantic_targets_covered']} of {r['references']}"
    q={}
    for name,ds,sid,task in [('FableQuestion',prose,'fable','actions'),('HarborQuestion',compact,'harbor','locations'),('OrchardQuestion',compact,'orchard','possession'),('OrchardBeliefQuestion',compact,'orchard','belief'),('AliceActionQuestion',prose,'alice','actions'),('AliceClaimQuestion',prose,'alice','claims')]:
        q[name]=next(r['question'] for r in ds['results'] if r['story_id']==sid and r['task']==task)
    (HERE/'generated/numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+tex(v)+'}' for k,v in {**numbers,**q}.items())+'\n')
    write_json(HERE/'generated/numerical_claims.json',numbers)
    rows=[]
    for r in cm:
        vals=[r['story'].title(),{'locations':'Location','possession':'Possession','belief':'Belief/reality'}[r['question']]]
        for m in ['A','B']:
            vals += [str(r[m+'_predicted'])]+[f"{r[m+'_'+v]:.3f}" for v in ['precision','recall','f1']]
        rows.append(' & '.join(vals)+r' \\')
    (HERE/'tables/compact.tex').write_text(r'''\begin{table*}[t]
% Positive inset keeps the top caption within the nominal text margin.
\vspace*{6pt}
\caption{Compact-story v2 strict qualified-fact scores. A: pre-extract/select. B: contextual extraction. $n$ includes every prediction. Targets per story: possession 5, location 4, belief/reality 4. No historical recoveries are pooled.}\label{tab:compact}
\centering\small
\begin{tabular}{llrrrrrrrr}\toprule
Story & Question & $n_A$ & $P_A$ & $R_A$ & $F1_A$ & $n_B$ & $P_B$ & $R_B$ & $F1_B$\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table*}'+'\n')
    rows=[]
    for r in pm:
        vals=[r['story'].title(),r['task'].title(),r['approach'][0]]
        if r['metric_status']=='scored':
            vals += [str(r['predictions'])]+[f"{r['strict_qualified_'+v]:.3f}" for v in ['precision','recall','f1']]
            vals += ['/'.join(str(r[k]) for k in ['semantically_supported','partially_supported','unsupported','unresolved']),f"{r['semantic_targets_covered']}/{r['references']}"]
        else: vals += ['NA']*6
        rows.append(' & '.join(vals)+r' \\')
    (HERE/'tables/prose.tex').write_text(r'''\begin{table*}[t]
\vspace*{6pt}
\caption{Published prose: strict agreement and Codex-authored semantic assessment. A: pre-extract/select. B: contextual extraction. S/P/X/U counts supported, partial, unsupported and unresolved records. Coverage counts preserved target meanings. NA denotes parse failure, not an empty graph. All parseable predictions cite valid evidence IDs, which does not establish support.}\label{tab:prose}
\centering\small\begin{tabular}{lllrrrrrr}\toprule
Passage & Question & Method & $n$ & Precision & Recall & F1 & S/P/X/U & Coverage\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table*}'+'\n')
    for name,dat in [('compact',cm),('prose',pm),('cost',alloc[2:])]:write_json(HERE/f'tables/{name}.json',dat)
    return numbers


def retained_requests(datasets):
    target=HERE/'data/requests.json'
    if target.exists():
        requests=read(target)
    else:
        requests={}
        patterns={'compact':'compact-story-v2-backup.*/run-*/request-*.json',
                  'prose':'real-text-poc-backup.*/run-*/request-*.json'}
        allowed={'best_of','chat_template_kwargs','frequency_penalty','ignore_eos','max_tokens','messages','min_p','model','n','presence_penalty','repetition_penalty','seed','stop_token_ids','stream','stream_options','temperature','top_k','top_p','use_beam_search'}
        for name,ds in datasets.items():
            requests[name]={}
            for f in (ROOT/'artifacts/restricted').glob(patterns[name]):
                payload=read(f)
                assert set(payload)==allowed
                h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                for call in ds['calls']:
                    if h==call['request_hash']:requests[name][call['case_id']]=payload
        write_json(target,requests)
    for name,ds in datasets.items():
        for c in ds['calls']:
            p=requests[name][c['case_id']]
            h=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            assert h==c['request_hash']
            assert p['seed']==1988649846 and p['max_tokens']==2048
    return requests


def companion(datasets, requests):
    """Build selected reader aids only. Full retained machine records are not rewritten."""
    from network_figures import load_panels
    panels = load_panels(datasets)
    parts = []
    def macro(name, value):
        parts.append('\\newcommand{\\'+name+'}{'+tex(value)+'}')
    for name,key in [('OrchardA','orchard_A'),('OrchardB','orchard_B'),('OrchardBelief','orchard_belief')]:
        ev=panels[key]['result']['evaluation']['full']
        macro(name+'Score',f"{ev['true_positive']}/{ev['predicted']} matched, {ev['reference_count']} targets. "
              f"P/R/F1 = {ev['precision']:.3f}/{ev['recall']:.3f}/{ev['f1']:.3f}.")
    real=datasets['prose']
    # Literal strings and frozen assessments are copied, never reinterpreted.
    for name,story,eid in [('FableOne','fable','S1'),('FableTwo','fable','S2'),
                           ('FableFour','fable','S4'),('FableFive','fable','S5'),
                           ('AliceOne','alice','S1')]:
        macro(name,real['stories'][story]['evidence'][eid])
    b=panels['fable_B_network']['result']
    for name,index in [('PassiveExample',2),('PartialExample',1)]:
        f=b['parsed']['facts'][index-1]
        a=b['semantic_assessments'][index-1]
        macro(name,' | '.join([f['subject'],f['relation'],f['object'],','.join(f['evidence_ids'])]))
        macro(name+'Assessment',a['note'])
    ev=b['evaluation']['full']
    macro('FableBScore',f"P/R/F1 = {ev['precision']:.3f}/{ev['recall']:.3f}/{ev['f1']:.3f}.")
    a=panels['fable_A_network']['result']['evaluation']['full']
    macro('FableAScore',f"P/R/F1 = {a['precision']:.3f}/{a['recall']:.3f}/{a['f1']:.3f}.")
    # Truncated DISPLAY excerpt of a complete, historically unparseable response.
    # It is never parsed as a recovered graph.
    c=next(c for c in real['calls'] if c['case_id']=='8')
    (HERE/'generated/holmes_tail.txt').write_text(c['raw_text'][-200:])
    macro('HolmesParseError',str(c['parse_error']))
    (HERE/'generated/companion_facts.tex').write_text('\n'.join(parts)+'\n')
    paths=list((HERE/'data').glob('*'))+[
        ROOT/'reports/tables/compact_story_v2_results.json',
        ROOT/'reports/tables/real_text_proof_of_concept.json']
    write_json(HERE/'generated/reproducibility_index.json',{
        'historical_companion_commit':'48c0def1bfe8880029d88e9539462302d77833c7',
        'historical_companion_pages':36,
        'complete_evidence_files':{str(p.relative_to(ROOT)):digest(p) for p in paths if p.is_file()},
        'requests':'data/requests.json, dataset key compact/prose, then case ID',
        'raw_responses':'data/retained_outputs.json, dataset, calls, raw_text',
        'reference_alternatives':'data/retained_outputs.json, dataset, references',
        'selected_records':'data/retained_outputs.json, dataset, results, parsed.facts',
        'semantic_assessments':'reports/tables/real_text_proof_of_concept.json, results[].semantic_assessments',
        'mappings':'manifest.json, figures[].records: exact fact, source call and record index',
        'no_evidence_removed':True})

def main():
    # Editable conference manuscripts are inputs, never generated from an older draft.
    roots = {'submission':'ICAART2027_submission.tex', 'companion':'ICAART2027_companion.tex'}
    before = {p:digest(HERE/p) for p in [*roots.values(), 'companion_source.tex', 'figure_blocks.tex']}
    for folder in ['figures','tables','generated','data']:(HERE/folder).mkdir(exist_ok=True)
    _,data,panels,hashes=graph.load_inputs()
    datasets={'compact':data['synthetic'],'prose':data['real']}
    req=retained_requests(datasets)
    nums=numbers_and_tables(**datasets)
    figs=figures(datasets,panels)
    from network_figures import load_panels, network_assets
    figs.update(network_assets(sys.modules[__name__], datasets, load_panels(datasets)))
    companion(datasets,req)
    assert all(digest(HERE/p)==h for p,h in before.items()), 'Editable manuscript changed during asset generation'
    # The audited conference bibliography is editable, not overwritten from the
    # historical general manuscript. Official apalike files remain byte-identical.
    write_json(HERE/'manifest.json',dict(source_editorial_commit='dafabd0fd5258b68965ec4b11a870ac5773d66d6',
               source_hashes={**hashes,**{p:digest(ROOT/p) for p in ['paper/manuscript_source.md','paper/AUTHOR_REVIEW.md','paper/FIGURE_CAPTIONS.md','paper/references.bib','paper/tables/numerical_registry.json','paper/tables/compact_comparison.json','paper/tables/published_comparison.json','paper/tables/allocation.json']}},
               template_archive_sha256=digest(HERE/'vendor/SCITEPRESS_Conference_Latex.zip'),
               template_files={p:digest(HERE/p) for p in ['article.cls','SCITEPRESS.sty','apalike.sty','apalike.bst']},
               conference_bibliography_sha256=digest(HERE/'references.bib'),
               manuscript_roots=roots,
               reference_audit_sha256=digest(HERE/'REFERENCE_AUDIT.md'),
               figures=figs,numerical_claims=nums,
               rendering_source_hashes={p:digest(HERE/p) for p in ['build_assets.py','network_figures.py','companion_source.tex','figure_blocks.tex']},
               main_figures=['orchard_network','fable','alice'],
               supplementary_figures=['orchard_comparison','orchard_beliefs','harbor','fable_networks'],
               evidence_preservation=read(HERE/'generated/reproducibility_index.json'),
               regeneration='python paper/icaart2027/build_assets.py && sh paper/icaart2027/build.sh && python paper/icaart2027/verify.py',
               selection_rationale={'fable':'Simple actions, conditional-scope loss and a comparison favouring A.',
                                    'orchard':'User-selected illustrative retained call 2: actual connected query slice. Supplement contrasts separate call 7 and the belief/reality selection.',
                                    'harbor':'Source intervals versus question window with all B relevance errors visible.',
                                    'alice':'Useful thought content beside a wrong participant and unresolved interpretation.'}))
    print('Conference networks, unchanged tables and selected reader aids prepared. Complete evidence files retained.')


if __name__=='__main__':main()
