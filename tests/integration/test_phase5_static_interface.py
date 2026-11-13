from __future__ import annotations

from pathlib import Path


def test_page_exposes_full_diff_and_progressive_detail_contract() -> None:
    root = Path(__file__).resolve().parents[2] / "ui"
    html = (root / "index.html").read_text(encoding="utf-8")
    javascript = (root / "app.js").read_text(encoding="utf-8")
    for marker in ("added", "removed", "merged", "split", "requalified"):
        assert f'class="swatch {marker}"' in html
    assert "comparisonBundle" in javascript
    assert "before-state" in javascript
    assert "Provenance" in javascript
    assert "Event roles" in javascript
    assert "Contextual relevance" in javascript
    assert "Holder-relative time" in javascript
    assert "Assertion support" in javascript
    assert "Description support" in javascript
    assert "Verified description evidence" in javascript
    assert "Projection-claimed description evidence (not verified)" in javascript
    assert "rendererOnly" in javascript
    assert "revisionSeedDecimal" in javascript
    assert "seed: 0" not in javascript
    assert "beforeProjectionId" in javascript


def test_page_states_strict_claim_boundary() -> None:
    html = (Path(__file__).resolve().parents[2] / "ui" / "index.html").read_text(encoding="utf-8")
    assert "local single-user research demonstration" in html.lower()
    assert "makes no" in html
    for forbidden_claim in ("improves usability", "reduces workload", "increases trust"):
        assert forbidden_claim not in html.lower()
