"""Concrete retained-record regressions for the conference network displays."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('icaart_networks', ROOT/'paper/icaart2027/network_figures.py')
net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net)


def panels():
    return net.load_panels({
        'compact': json.loads((ROOT/'reports/tables/compact_story_v2_results.json').read_text()),
        'prose': json.loads((ROOT/'reports/tables/real_text_proof_of_concept.json').read_text()),
    })


def test_orchard_selection_is_actual_call_subset():
    p = panels()
    assert net.panel_counts(p['orchard_pre']) == (10,14)
    assert net.panel_counts(p['orchard_A']) == (6,5)
    assert [r['source_indices'] for r in p['orchard_A']['records']] == [[1],[2],[3],[4],[13]]
    for r in p['orchard_A']['records']:
        assert r['fact'] == p['orchard_pre']['records'][r['source_indices'][0]-1]['fact']


def test_contextual_output_remains_separate_and_keeps_extras():
    p = panels()
    b = p['orchard_B']
    assert b['source_call'] == '7'
    assert net.panel_counts(b) == (10,12)
    assert sum(r['status']=='irrelevant' for r in b['records']) == 7
    assert b['result']['evaluation']['full']['predicted'] == 12


def test_beliefs_are_edge_qualifications_not_invented_nodes():
    p = panels()['orchard_belief']
    assert net.panel_counts(p) == (6,4)
    facts = [r['fact'] for r in p['records']]
    assert sum(f['attitude']=='believes' for f in facts) == 2
    endpoints = {f[k] for f in facts for k in ('subject','object')}
    assert not {'Jun','Soren'} & endpoints
    assert all(f['valid_from'] is None and f['valid_until'] is None for f in facts)


def test_fable_parallel_edges_and_literal_identities_survive():
    p = panels()
    a = p['fable_A_network']
    b = p['fable_B_network']
    assert net.panel_counts(a) == (5,8)
    assert net.panel_counts(b) == (4,5)
    assert sum(r['fact']['subject']=='A LION' and r['fact']['object']=='a Mouse'
               for r in a['records']) == 4
    assert {'strong ropes','the rope'} <= {r['fact']['object'] for r in a['records']}
    spare = a['records'][2]
    assert spare['status'] == 'partial'
    assert 'valid_until' not in spare['fact']
    assert 'valid until' in spare['fact']
