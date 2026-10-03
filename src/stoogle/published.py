"""Import a StatLean commit from its published doc-gen4 HTML pages.

Every module in the commit must have a page whose source links point to that commit.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
from bs4 import BeautifulSoup

from stoogle.corpus import git, pinned_commit, plain, source_identity, source_versions, write_corpus

KINDS = {"theorem", "def", "definition", "structure", "class", "inductive", "opaque", "abbrev"}


def source_modules(repo, commit):
    files = git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
    return {p[:-5].replace("/", "."): p for p in files
            if p.endswith(".lean") and (p == "StatLean.lean" or p.startswith("StatLean/"))}


def parse_page(html, module, path, repo, commit, origin, curated):
    soup = BeautifulSoup(html, "html.parser")
    links = [a["href"] for a in soup.select('a[href]')
             if a["href"].startswith("https://github.com/") and "/blob/" in a["href"]]
    expected = (origin, commit, path)
    if not links or any(source_identity(url) != expected for url in links):
        raise ValueError(f"Documentation for {module} was not built from commit {commit}")
    source = git(repo, "show", f"{commit}:{path}").splitlines(keepends=True)
    records, excluded = [], Counter()
    for decl in soup.select("div.decl[id]"):
        name = decl["id"]
        header = decl.select_one(".decl_header")
        kind_node = header.select_one(".decl_kind") if header else None
        if kind_node is None:
            raise ValueError(f"Unknown declaration HTML for {name}")
        kind = kind_node.get_text(strip=True).removeprefix("noncomputable ").removeprefix("unsafe ")
        if kind not in KINDS | {"instance", "constructor"}:
            raise ValueError(f"Unsupported declaration kind {kind!r}: {name}")
        if kind not in KINDS or name.startswith("_private.") or name.endswith(".mk"):
            excluded[kind] += 1
            continue
        link = decl.select_one(".gh_link a[href]")
        match = re.fullmatch(r"L(\d+)-L(\d+)", urlsplit(link["href"]).fragment) if link else None
        if not match:
            raise ValueError(f"Missing source range for {name}")
        start, end = map(int, match.groups())
        if not 1 <= start <= end <= len(source):
            raise ValueError(f"Out-of-bounds source range for {name}")
        metadata = curated.get(name, {})
        # doc-gen4 renders Markdown immediately after the header, without a
        # .docstring wrapper. Do not include structure fields/equations as prose.
        docstring = "\n\n".join(node.get_text().strip() for node in header.next_siblings
                                if getattr(node, "name", None) in {"p", "ul", "ol", "pre", "blockquote", "table", "h1", "h2", "h3", "h4"})
        refs = {unquote(urlsplit(a["href"]).fragment) for a in header.select("a[href]")}
        records.append({
            "id": "sl_" + hashlib.sha256((module + "::" + name).encode()).hexdigest()[:20],
            "name": name, "module": module,
            "kind": "definition" if kind in {"def", "definition", "abbrev"} else kind,
            "signature": re.sub(r"\n\s*\n", "\n", header.get_text().strip()),
            "source_text": "".join(source[start-1:end]),
            "source_link": link["href"], "file": path, "line": start,
            "import": f"import {module}", "docstring": docstring,
            "title": plain(metadata.get("title", "")),
            "description": plain(metadata.get("summary") or docstring),
            "description_origin": "website" if metadata.get("summary") else ("doc-gen4-html" if docstring else "none"),
            "citation": plain(metadata.get("citation", "")), "category": metadata.get("category", ""),
            "featured": bool(metadata), "website_id": metadata.get("id"),
            "signature_dependencies": sorted(refs - {name, ""}),
            # The HTML pages do not record whether a proof uses sorry.
            "proof_status": "not_audited", "commit": commit, "repository": origin,
        })
    return records, excluded


def export_published(repo: Path, commit: str | None, base: str, cache: Path, output: Path,
                     origin="https://github.com/StatLean/Stat-Lean", workers=4, tag: str | None = None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    commit = pinned_commit(repo, commit, tag)
    if urlsplit(base).scheme != "https" or urlsplit(base).hostname != "statlean.github.io":
        raise ValueError("The documentation base must be an https://statlean.github.io URL")
    if not 1 <= workers <= 8:
        raise ValueError("Workers must be between 1 and 8")
    modules = source_modules(repo, commit)
    if not modules:
        raise ValueError(f"No StatLean modules at commit {commit}")
    curated = {r["fullName"]: r for r in json.loads(git(repo, "show", f"{commit}:website/src/data/results.json"))}
    cache = cache / commit
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        module, path = item
        destination = cache / (module + ".html")
        url = base.rstrip("/") + "/" + path[:-5] + ".html"
        try:
            if destination.exists():
                html = destination.read_text(encoding="utf-8")
            else:
                for attempt in range(3):
                    try:
                        response = httpx.get(url, timeout=45, follow_redirects=False)
                        response.raise_for_status()
                        html = response.text
                        break
                    except httpx.HTTPError:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            records, excluded = parse_page(html, module, path, repo, commit, origin, curated)
            if not destination.exists():
                destination.write_text(html, encoding="utf-8")
            return module, records, excluded, hashlib.sha256(html.encode()).hexdigest(), None
        except (ValueError, httpx.HTTPError) as error:
            return module, [], Counter(), None, str(error)

    records, excluded, errors, digests = [], Counter(), {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for module, found, omitted, digest, error in pool.map(fetch, sorted(modules.items())):
            if error:
                errors[module] = error
            else:
                records.extend(found)
                excluded.update(omitted)
                digests[module] = digest
    coverage = {"expected_modules": len(modules), "documented_modules": len(digests),
                "missing_or_invalid_modules": errors, "module_html_sha256": digests,
                "excluded_declarations_by_kind": dict(excluded),
                "declaration_counts": dict(Counter(r["module"] for r in records)),
                "complete": not errors}
    names = Counter(r["name"] for r in records)
    conflicts = {name: [r["module"] for r in records if r["name"] == name]
                 for name, count in names.items() if count > 1}
    coverage["same_name_in_multiple_modules"] = conflicts
    for record in records:
        if record["name"] in conflicts:
            record["name_conflict_modules"] = conflicts[record["name"]]
    if errors:
        output.mkdir(parents=True)
        (output / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
        raise ValueError(f"Coverage failed: {len(errors)}/{len(modules)} modules; see {output}/coverage.json. No corpus published.")
    manifest = {"schema_version": 1, "repository": origin, "commit": commit,
                **source_versions(repo, commit, tag),
                "module_prefixes": ["StatLean"], "declaration_count": len(records),
                "featured_count": sum(r["featured"] for r in records),
                "documentation_origin": base, "coverage_complete": True,
                "module_count": len(modules),
                "name_conflict_count": len(conflicts),
                "coverage_sha256": hashlib.sha256(json.dumps(coverage, sort_keys=True).encode()).hexdigest()}
    write_corpus(output, records, manifest)
    (output / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
    return manifest
