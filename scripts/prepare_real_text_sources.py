"""Extract three preselected contiguous public-domain excerpts without changing their text."""

import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from story_projection_onto.manifest import write_json_atomic


def prepare(root):
    source = root / "artifacts/restricted/published-prose-sources"
    destination = root / "data/published_prose"
    destination.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "fable",
            21,
            "Three hundred Aesop’s fables",
            "Aesop",
            "George Fyler Townsend",
            "The Lion And The Mouse",
            "A LION was awakened",
            "\r\n\r\n\r\n\r\nThe Wolf And The Lamb",
            ["face.", "kindness.”", "let him go.", "the ground.", "exclaiming:"],
        ),
        (
            "alice",
            11,
            "Alice’s Adventures in Wonderland",
            "Lewis Carroll",
            None,
            "Chapter I: Down the Rabbit-Hole, opening two paragraphs",
            "Alice was beginning",
            "\r\n\r\nThere was nothing so",
            ["conversations?”"],
        ),
        (
            "holmes",
            1661,
            "The Adventures of Sherlock Holmes",
            "Arthur Conan Doyle",
            None,
            "II. The Red-Headed League, opening visit through the gentleman’s greeting",
            " I had called upon my friend, Mr. Sherlock Holmes",
            "\r\n\r\n“Try the settee,”",
            [
                "fiery red hair.",
                "the door behind me.",
                "said cordially.",
                "were engaged.”",
                "Very much so.”",
                "next room.”",
                "yours also.”",
            ],
        ),
    ]
    result = {}
    for sid, ebook, title, author, translator, section, begin, end, boundaries in specs:
        raw = (source / f"pg{ebook}.txt").read_bytes()
        text = raw.decode("utf-8-sig")
        start = text.index(begin)
        stop = text.index(end, start)
        excerpt = text[start:stop]
        cuts, cursor = [0], 0
        for marker in boundaries:
            cursor = excerpt.index(marker, cursor) + len(marker)
            cuts.append(cursor)
        cuts.append(len(excerpt))
        spans = {f"S{i}": excerpt[a:b] for i, (a, b) in enumerate(pairwise(cuts), 1)}
        assert "".join(spans.values()) == excerpt
        notice = text[: text.index("*** START OF THE PROJECT GUTENBERG")]
        license_start = text.index("*** END OF THE PROJECT GUTENBERG")
        notices = notice + "\r\n" + text[license_start:]
        notice_path = destination / f"pg{ebook}_notices.txt"
        if notice_path.exists():
            assert notice_path.read_bytes() == notices.encode()
        else:
            notice_path.write_bytes(notices.encode())
        result[sid] = dict(
            title=title,
            author=author,
            translator=translator,
            section=section,
            source_url=f"https://www.gutenberg.org/ebooks/{ebook}",
            download_url=f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt",
            retrieval_date=datetime.fromtimestamp(
                (source / f"pg{ebook}.txt").stat().st_mtime, UTC
            ).isoformat(),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
            excerpt=excerpt,
            evidence=spans,
            word_count=len(excerpt.split()),
            notice_file=f"data/published_prose/{notice_path.name}",
            notice_sha256=hashlib.sha256(notices.encode()).hexdigest(),
            rights="Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA.",
            segmentation="Exact contiguous UTF-8 excerpt, including original whitespace; concatenated evidence spans reproduce it byte for byte. No narrative rewriting.",
        )
    path = destination / "passages.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        write_json_atomic(result, path)
    print({k: {a: v[a] for a in ("word_count", "excerpt_sha256")} for k, v in result.items()})


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
