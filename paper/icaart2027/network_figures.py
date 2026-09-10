"""Small, deterministic network views of retained records. No inference or scoring."""
from __future__ import annotations

import math

from matplotlib.path import Path

def result_panel(dataset, story, task, method, key):
    r = next(r for r in dataset['results'] if r['story_id'] == story
             and r['task'] == task and r['approach'].startswith(method))
    c = next(c for c in dataset['calls'] if c['case_id'] == r['source_case_id'])
    records = []
    for i, f in enumerate(r['parsed']['facts'], 1):
        if 'semantic_assessments' in r:
            a = r['semantic_assessments'][i-1]
            state = 'unsupported' if a['support'] == 'contradicted' else a['support']
            if state == 'supported' and not a['relevant']:
                state = 'irrelevant'
            note = a['note']
        else:
            a = r['evaluation']['rows'][i-1]
            state = {'correct_complete_fact': 'supported',
                     'supported_but_irrelevant': 'irrelevant',
                     'unsupported': 'unsupported'}.get(a['status'], 'unresolved')
            note = a['status']
        source_indices = [j for j, cf in enumerate(c['parsed']['facts'], 1) if cf == f]
        assert source_indices
        records.append(dict(index=i, fact=f, status=state, note=note,
                            source_indices=source_indices))
    return dict(id=key, dataset='synthetic' if story == 'orchard' else 'real',
                story=story, records=records, result=r, source_call=c['case_id'],
                original_count=len(records), omitted_count=0,
                request_hash=c['request_hash'], response_sha256=c['response']['response_sha256'])


def load_panels(data):
    ds = data['compact']
    panels = {name: result_panel(ds, 'orchard', task, method, name)
              for name, task, method in [('orchard_A', 'possession', 'A'),
                                         ('orchard_B', 'possession', 'B'),
                                         ('orchard_belief', 'belief', 'A')]}
    c = next(c for c in ds['calls'] if c['case_id'] == '2')
    # No query-specific status is retrospectively assigned to the broad extraction.
    panels['orchard_pre'] = dict(id='orchard_pre', dataset='synthetic', story='orchard',
        source_call='2', result=None, original_count=len(c['parsed']['facts']), omitted_count=0,
        request_hash=c['request_hash'], response_sha256=c['response']['response_sha256'],
        records=[dict(index=i, fact=f, status='not_assessed_in_this_view',
                      note='Literal model pre-extraction, not a reference graph.', source_indices=[i])
                 for i, f in enumerate(c['parsed']['facts'], 1)])
    for key, counts in [('orchard_pre', (10,14)), ('orchard_A', (6,5)), ('orchard_B', (10,12))]:
        assert panel_counts(panels[key]) == counts, (key, panel_counts(panels[key]))
    assert [r['source_indices'] for r in panels['orchard_A']['records']] == [[1],[2],[3],[4],[13]]
    for key in ['fable_A_network', 'fable_B_network']:
        panels[key] = result_panel(data['prose'], 'fable', 'actions', key[6], key)
    return panels


def panel_counts(panel):
    return (len({r['fact'][k] for r in panel['records'] for k in ('subject', 'object')}),
            len(panel['records']))


# Author-chosen display positions and categories, NOT model-generated types.
ORCHARD = {'River Room': (54,30), 'Copper Book': (204,60), 'South Loft': (381,30),
           'Jun': (55,157), 'Vera': (307,136), 'Ivory Vase': (260,260),
           'Soren': (410,166), 'Hill Court': (406,310),
           'Elm Dock': (166,316), 'Jade Drum': (48,275)}
CATEGORIES = {n: ('person' if n in ('Vera','Jun','Soren') else
                  'object' if n in ('Copper Book','Ivory Vase','Jade Drum') else 'place')
              for n in ORCHARD}
# (arc curvature, label x, label y). Keys are cited source IDs, not expected facts.
ORCHARD_ROUTES = {
    'S1': (0,263,77), 'S2': (0,351,215), 'S3': (0,132,103),
    'S4': (0,269,177), 'S5': (0,40,87), 'S6': (0,412,231),
    'S7': (0,370,95), 'S8': (0,104,295), 'S9': (-.10,299,12),
    'S10': (0,133,26), 'S11': (0,337,286), 'S12': (0,215,290),
    'S13': (0,43,210), 'S14': (.45,209,197),
}


def draw_network(b, p, panel, origin, positions, routes, *, yscale=1, categories=None,
                 broad=False, local_id=None):
    """One glyph per exact endpoint string. Each retained record gets its own arrow."""
    g = b.graph
    ox, oy = origin
    actual = {r['fact'][k] for r in panel['records'] for k in ('subject','object')}
    assert actual <= positions.keys()
    boxes, node_meta, box_sizes = {}, [], {}
    for name in sorted(actual):
        x,y = positions[name]; x+=ox; y=oy+y*yscale
        cat = (categories or {}).get(name, 'endpoint')
        fc = {'person':'#E4EDF5', 'object':'#FFF1D4', 'place':'#F1F1F1'}.get(cat,'#F1F5F7')
        lines = g.wrap(name, 69, 8.5)
        width = min(80, max(45, max(b.conference_text_width(t,8.5) for t in lines)+10))
        height = len(lines)*11.05+9
        boxes[name] = p.rect(x-width/2,y-height/2,width,height,fc=fc,ec='#566573',radius=3)
        boxes[name].set_zorder(4)
        box_sizes[name]=(width,height)
        p.text(x,y-len(lines)*11.05/2,'\n'.join(lines),size=8.5,align='center')
        node_meta.append(dict(identity=name, x=x, y=y, category=cat))
    for r in panel['records']:
        f=r['fact']; sx,sy=positions[f['subject']]; tx,ty=positions[f['object']]
        route_key = f['evidence_ids'][0] if panel['story']=='orchard' else r['index']
        rad,lx,ly = routes.get(route_key,(0,(sx+tx)/2,(sy+ty)/2-10))
        if panel['story']=='fable':
            rad*=yscale
        if broad:
            color,symbol,style = '#384B59','',('dashed' if f.get('holder') else 'solid')
        else:
            color,symbol,style = g.STATUS[r['status']]
        geometry=dict(kind='arc3',curvature=rad)
        if panel['story']=='orchard' and route_key=='S14':
            # Two explicitly routed quadratic segments avoid passing through Vera
            # or Ivory Vase. These are display waypoints, never graph vertices.
            def xy(x,y):return (ox+x,oy+y*yscale)
            def clipped(name, toward):
                cx,cy=xy(*positions[name]); w,h=box_sizes[name]
                dx,dy=toward[0]-cx,toward[1]-cy
                ratio=min(w/2/abs(dx) if dx else float('inf'),
                          h/2/abs(dy) if dy else float('inf'))
                ratio+=2/math.hypot(dx,dy)
                return (cx+dx*ratio,cy+dy*ratio)
            first,mid,last=xy(200,220),xy(337,189),xy(418,160)
            vertices=[clipped(f['subject'],first),first,mid,last,clipped(f['object'],last)]
            path=Path(vertices,[Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3])
            arrow=g.FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=10,
                lw=1.15,color=color,linestyle=style,zorder=2)
            geometry=dict(kind='two_quadratics',vertices=vertices,
                          note='Routing waypoints are not nodes. Endpoints clipped to actual node borders.')
        else:
            arrow = g.FancyArrowPatch((ox+sx,oy+sy*yscale),(ox+tx,oy+ty*yscale),
                connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',mutation_scale=10,
                lw=1.15,color=color,linestyle=style,patchA=boxes[f['subject']],
                patchB=boxes[f['object']],shrinkA=2,shrinkB=2,zorder=2)
        p.ax.add_patch(arrow)
        relation=f['relation'].replace('_',' ')
        if relation=='located in':relation='loc'
        label=(symbol+' '+relation).strip()+' · '+','.join(f['evidence_ids'])
        if f.get('valid_from') is not None or f.get('valid_until') is not None:
            label+=f"\n[{f.get('valid_from')},{f.get('valid_until')})"
        if f.get('holder') is not None or f.get('attitude') is not None:
            label+=f"\n{f.get('holder')} / {f.get('attitude')}"
        if 'valid_until' not in f:label+='\nvalid_until MISSING'
        text=p.text(ox+lx,oy+ly*yscale,label,size=8.5,align='center',color=color)
        text.set_bbox(dict(facecolor='white',edgecolor='none',alpha=.97,pad=.8))
        g.register_edge(p,panel,r,[arrow,text])
        p.edges[-1]['source_call']=panel['source_call']
        p.edges[-1]['source_indices']=r['source_indices']
        p.edges[-1]['display_label']=label
        p.edges[-1]['display_route']=geometry
    if not hasattr(p,'network_panels'):p.network_panels=[]
    p.network_panels.append(dict(id=local_id or panel['id'],source_panel=panel['id'],
        nodes=node_meta,node_count=len(actual),edge_count=len(panel['records']),
        indices=[r['index'] for r in panel['records']],omitted_record_count=0,
        identity_rule='Exact endpoint string equality only. No holder-generated edges.',
        position_rule='Declared fixed display coordinates; vertical scale recorded.',yscale=yscale))


def network_assets(b, data, panels):
    out={}
    p=b.make_plate('orchard_network')
    p.text(5,2,'Orchard · text, model network and fixed query selection',size=10,bold=True)
    y=20
    for eid in ['S1','S3','S11']:
        y=p.para(5,y,eid+' '+data['compact']['stories']['orchard']['evidence'][eid],b.W-10,size=8.5)+2
    p.text(5,y+4,'Model extraction before the question · 10 nodes / 14 records',size=9,bold=True)
    draw_network(b,p,panels['orchard_pre'],(0,y+28),ORCHARD,ORCHARD_ROUTES,yscale=.70,categories=CATEGORIES,broad=True)
    y+=270
    p.text(5,y,'Fixed selection: ownership/carrying · 6 nodes / 5 records',size=9,bold=True)
    # Same x positions and order, less vertical whitespace in the sparse query view.
    sparse_routes={**ORCHARD_ROUTES, 'S4':(0,252,150), 'S13':(0,43,190)}
    draw_network(b,p,panels['orchard_A'],(0,y+8),ORCHARD,sparse_routes,yscale=.35,categories=CATEGORIES)
    y+=124
    y=p.para(5,y,'Blue fill: person. Cream: object. Grey: place (display categories). loc = located in. '
             'Dashed broad edges carry holder / attitude, not extra relationships. + supported. '
             'Unprinted bounds and attribution are null. Source excerpted: all 14 sentences supplied.',b.W-10,size=8.5)+6
    out[p.name]=b.save_plate(p,y,'All 14 pre-extraction records and all five actually selected records. '
        'Only exact endpoint strings share glyphs. Broad graph is ungraded, not gold. '
        'Sparse panel compresses vertical spacing; x anchors remain fixed. Source excerpt S1,S3,S11.')

    p=b.make_plate('orchard_comparison')
    p.text(5,3,'Orchard ownership · fixed selection versus contextual extraction',size=10,bold=True)
    p.text(5,25,'A: five selected records from call 2',size=9,bold=True)
    draw_network(b,p,panels['orchard_A'],(0,25),ORCHARD,ORCHARD_ROUTES,yscale=.45,categories=CATEGORIES)
    p.text(5,188,'B: independent contextual call 7 · all twelve records',size=9,bold=True)
    draw_network(b,p,panels['orchard_B'],(0,228),ORCHARD,ORCHARD_ROUTES,yscale=.85,categories=CATEGORIES)
    y=p.para(5,510,'+ supported and relevant. I supported but irrelevant (dashed). '
        'Colours assess records, not selection. Node fills are display categories. '
        'loc = located in. All holder/attitude values null. Unprinted bounds null.',b.W-10,size=8.5)+7
    out[p.name]=b.save_plate(p,y,'Complete separate A and B possession outputs. No combined graph. '
        'Common x coordinates and relative y order. All seven B extras remain visible and scored.')

    p=b.make_plate('orchard_beliefs')
    p.text(5,3,'Orchard · believed and narrated locations',size=10,bold=True)
    y=24
    for eid in ['S9','S10','S11','S12']:
        y=p.para(5,y,eid+' '+data['compact']['stories']['orchard']['evidence'][eid],b.W-10,size=9)+5
    positions={'Copper Book':(80,90),'South Loft':(325,32),'River Room':(325,148),
               'Ivory Vase':(80,287),'Hill Court':(325,228),'Elm Dock':(325,345)}
    routes={'S9':(0,203,43),'S10':(0,203,120),'S11':(0,203,239),'S12':(0,203,315)}
    draw_network(b,p,panels['orchard_belief'],(0,y),positions,routes,categories=CATEGORIES)
    y=p.para(5,y+381,'A: fixed selection from call 2. All four records on six endpoint nodes. '
        'Holder names are qualifications, not additional nodes or edges. All validity bounds null. '
        '+ supported under the frozen evaluation. loc = located in.',b.W-10,size=8.5)+6
    out[p.name]=b.save_plate(p,y,'Complete belief/reality selection. Two disconnected object-centred components. '
        'Explicit holder/attitude preserved on two attributed edges. No invented holder edges.')

    p=b.make_plate('fable_networks')
    p.text(5,3,'Fable actions · full networks with literal endpoint identities',size=10,bold=True)
    p.text(5,24,'A: fixed selection · all eight records',size=9,bold=True)
    pos={'A LION':(113,168),'a Mouse':(335,168),'some hunters':(57,42),
         'strong ropes':(62,299),'the rope':(365,299)}
    routes={1:(-.9,221,56),2:(-.40,219,109),3:(0,227,151),4:(.35,219,202),
            5:(0,81,83),6:(0,71,240),7:(0,365,241),8:(-.8,232,259)}
    draw_network(b,p,panels['fable_A_network'],(0,34),pos,routes,yscale=.8)
    p.text(5,309,'B: independent contextual extraction · all five records',size=9,bold=True)
    pos={'Lion':(115,105),'Mouse':(340,105),'hunters':(46,215),'ropes':(313,235)}
    routes={1:(.38,225,44),2:(0,64,146),3:(0,209,176),4:(0,344,166),5:(.4,225,146)}
    draw_network(b,p,panels['fable_B_network'],(0,311),pos,routes,yscale=.78)
    y=p.para(5,513,'+ supported meaning. ~ partial. ? unresolved. Exact endpoint strings remain separate: '
        'A has “strong ropes” and “the rope”, whereas B uses “ropes”. All attribution and bounds '
        'null except A’s missing valid_until field on spare (literal extra key “valid until”).',b.W-10,size=8.5)+7
    out[p.name]=b.save_plate(p,y,'All A and B action records, as separate networks. Parallel and reverse edges '
        'remain separate. No lexical aliases merged. A ambiguous awakening and conditional spare retained.')
    return out
