"""Evaluation queries from the website's informal statements.

The `informal` field is not part of the indexed text.
"""
import json
import random
import re
from pathlib import Path

from stoogle.corpus import git, read_corpus

MAX_QUERY_CHARS = 600


def informal_query(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())[:MAX_QUERY_CHARS]


def write_heldout(repo: Path, corpus: Path, output: Path, count=200, seed=0):
    records, manifest = read_corpus(corpus)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    names = [r["name"] for r in records]
    unique = {n for n in names if names.count(n) == 1}
    curated = json.loads(git(repo, "show", f"{manifest['commit']}:website/src/data/results.json"))
    cases = [{"query": informal_query(r["informal"]), "relevant": [r["fullName"]],
              "group": "heldout-informal", "website_id": r["id"]}
             for r in curated if r["fullName"] in unique and r.get("informal")]
    random.Random(seed).shuffle(cases)
    cases = cases[:count]
    if not cases:
        raise ValueError("No website results match this corpus")
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"output": str(output), "queries": len(cases), "commit": manifest["commit"], "seed": seed}
