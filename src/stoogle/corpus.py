"""Corpus files, source snapshot metadata, and import from doc-gen4's SQLite database.

RenderedCode decoding is adapted from LeanExplore's doc_parser.py at
6b25f8632cc3387cf85f7730375a690a1f1dfb79 (Apache-2.0); see NOTICE.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from urllib.parse import unquote, urlsplit


def plain(text: str) -> str:
    # Docstrings are Markdown; only whitespace is normalized, so text such as `i<n` is kept.
    return " ".join(text.split())


class RenderedCode:
    """Strict reader for leansqlite TaggedText; retains text and constant names."""
    def __init__(self, data: bytes):
        self.data, self.cursor, self.names = bytes(data), 0, set()

    def byte(self):
        if self.cursor >= len(self.data):
            raise ValueError("Truncated doc-gen4 RenderedCode")
        result = self.data[self.cursor]
        self.cursor += 1
        return result

    def nat(self):
        value = shift = 0
        while shift < 64:
            b = self.byte()
            value |= (b & 127) << shift
            if b < 128:
                return value
            shift += 7
        raise ValueError("Oversized RenderedCode integer")

    def string(self):
        size = self.nat()
        end = self.cursor + size
        if end > len(self.data):
            raise ValueError("Truncated RenderedCode string")
        text = self.data[self.cursor:end].decode("utf-8")
        self.cursor = end
        return text

    def name(self):
        tag = self.byte()
        if tag == 0:
            return ""
        if tag not in (1, 2):
            raise ValueError("Unknown Lean Name tag")
        parent = self.name()
        component = self.string() if tag == 1 else str(self.nat())
        return f"{parent}.{component}" if parent else component

    def walk(self):
        tag = self.byte()
        if tag == 0:
            return self.string()
        if tag == 2:
            return "".join(self.walk() for _ in range(self.nat()))
        if tag == 1:
            kind = self.byte()
            if kind == 2:
                self.names.add(self.name())
            elif kind not in (0, 1, 3, 4, 5, 6, 7):
                raise ValueError("Unknown RenderedCode tag")
            return self.walk()
        raise ValueError("Unknown TaggedText tag")

    def decode(self):
        text = self.walk()
        if self.cursor != len(self.data):
            raise ValueError("Trailing bytes in RenderedCode")
        return text, self.names


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    )


def resolve_tag(repo: Path, tag: str) -> str:
    """Full commit SHA of a tag in the source repository."""
    try:
        return git(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        raise ValueError(f"Tag {tag!r} not found in {repo}") from None


def pinned_commit(repo: Path, commit: str | None, tag: str | None) -> str:
    """The commit to import, given a full SHA, a tag, or both (which must agree)."""
    if tag:
        tagged = resolve_tag(repo, tag)
        if commit and commit != tagged:
            raise ValueError(f"Tag {tag} points to {tagged}, not {commit}")
        commit = tagged
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use a full 40-character Git commit or a tag")
    return commit


def source_versions(repo: Path, commit: str, tag: str | None = None) -> dict:
    """Commit date, StatLean tag, Lean toolchain and Mathlib revision of a source snapshot."""
    versions = {"commit_date": git(repo, "show", "-s", "--format=%cs", commit).strip(),
                "lean_toolchain": git(repo, "show", f"{commit}:lean-toolchain").strip()}
    try:
        packages = json.loads(git(repo, "show", f"{commit}:lake-manifest.json"))["packages"]
    except subprocess.CalledProcessError:
        packages = []
    mathlib = next((p for p in packages if p.get("name") == "mathlib"), None)
    if mathlib:
        versions["mathlib_rev"] = mathlib["rev"]
        if mathlib.get("inputRev"):
            versions["mathlib_input_rev"] = mathlib["inputRev"]
    if tag:
        versions["statlean_tag"] = tag
    return versions


def source_identity(url: str):
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.netloc != "github.com"
            or len(parts) < 5 or parts[2] != "blob"
            or not re.fullmatch(r"[0-9a-f]{40}", parts[3])):
        raise ValueError(f"doc-gen4 source URL must pin a GitHub commit: {url}")
    return "https://github.com/" + "/".join(parts[:2]), parts[3], unquote("/".join(parts[4:]))


def export_corpus(repo: Path, database: Path, prefixes: list[str], output: Path, tag: str | None = None):
    """Import declarations from a doc-gen4 SQLite database at the commit it records.

    The source repository must contain that commit; databases that mix commits are rejected.
    """
    repo = repo.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not prefixes or any(not (p == "StatLean" or p.startswith("StatLean.")) for p in prefixes):
        raise ValueError("Module prefixes must be inside StatLean")
    records, identities, source_cache, metadata_cache = [], set(), {}, {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        # A read transaction ensures all queries observe one doc-gen4 snapshot.
        db.execute("BEGIN")
        rows = db.execute("""
            SELECT n.*, r.start_line, r.start_column, r.end_line, r.end_column,
                   m.source_url, d.text AS docstring
            FROM name_info n
            JOIN modules m ON m.name = n.module_name
            JOIN declaration_ranges r ON r.module_name=n.module_name AND r.position=n.position
            LEFT JOIN declaration_markdown_docstrings d
                ON d.module_name=n.module_name AND d.position=n.position
            WHERE n.render=1 AND (n.module_name='StatLean' OR n.module_name LIKE 'StatLean.%')
            ORDER BY n.module_name, n.position
        """).fetchall()
        for row in rows:
            module, name = row["module_name"], row["name"]
            if not any(module == p or module.startswith(p + ".") for p in prefixes):
                continue
            if row["kind"] not in {"theorem", "def", "definition", "structure", "inductive", "opaque", "class"}:
                continue
            if name.startswith("_private.") or name.endswith(".mk"):
                continue
            origin, commit, path = source_identity(row["source_url"])
            identities.add((origin, commit))
            if len(identities) > 1:
                raise ValueError("Mixed source commits in doc-gen4 export; rebuild docs at one commit")
            if path not in source_cache:
                source_cache[path] = git(repo, "show", f"{commit}:{path}")
            source = source_cache[path]
            if commit not in metadata_cache:
                try:
                    curated = json.loads(git(repo, "show", f"{commit}:website/src/data/results.json"))
                except subprocess.CalledProcessError:
                    curated = []
                metadata_cache[commit] = {r["fullName"]: r for r in curated}
            metadata = metadata_cache[commit].get(name, {})
            result_type, names = RenderedCode(row["type"]).decode()
            binders = []
            for arg in db.execute("SELECT binder FROM declaration_args WHERE module_name=? AND position=? ORDER BY sequence", (module, row["position"])):
                binder, referenced = RenderedCode(arg["binder"]).decode()
                binders.append(binder)
                names.update(referenced)
            kind = "definition" if row["kind"] in ("def", "definition") else row["kind"]
            keyword = "def" if kind == "definition" else kind
            signature = f"{keyword} {name}" + "".join("\n    " + b for b in binders) + " :\n    " + result_type
            lines = source.splitlines(keepends=True)
            start, end = row["start_line"], row["end_line"]
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f"Source ranges out of bounds for {name}")
            selected = lines[start-1:end]
            # Lean source columns count Unicode code points; end is exclusive.
            selected[-1] = selected[-1][:row["end_column"]]
            selected[0] = selected[0][row["start_column"]:]
            docstring = row["docstring"] or ""
            description = plain(metadata.get("summary") or docstring)
            title = plain(metadata.get("title", ""))
            records.append({
                "id": "sl_" + hashlib.sha256(name.encode()).hexdigest()[:20],
                "name": name, "module": module, "kind": kind,
                "signature": signature, "source_text": "".join(selected),
                "source_link": row["source_url"] + f"#L{start}-L{end}",
                "file": path, "line": start, "import": f"import {module}",
                "docstring": docstring, "title": title, "description": description,
                "description_origin": "website" if metadata.get("summary") else ("docstring" if docstring else "none"),
                "citation": plain(metadata.get("citation", "")),
                "category": metadata.get("category", ""),
                "featured": bool(metadata), "website_id": metadata.get("id"),
                "signature_dependencies": sorted(names - {name, ""}),
                "proof_status": "contains_sorry" if row["sorried"] else "not_audited",
                "commit": commit, "repository": origin,
            })
        if not records:
            raise ValueError("No public declarations match the module prefixes")
        origin, commit = next(iter(identities))
        if tag:
            pinned_commit(repo, commit, tag)
        manifest = {
            "schema_version": 1, "repository": origin, "commit": commit,
            **source_versions(repo, commit, tag),
            "module_prefixes": prefixes, "declaration_count": len(records),
            "featured_count": sum(r["featured"] for r in records),
        }
    write_corpus(output, records, manifest)
    return manifest


def write_corpus(directory: Path, records: list[dict], manifest: dict):
    directory.mkdir(parents=True, exist_ok=False)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    (directory / "corpus.jsonl").write_text(payload, encoding="utf-8")
    manifest = {**manifest, "corpus_sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_corpus(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = (directory / "corpus.jsonl").read_bytes()
    if manifest.get("schema_version") != 1 or hashlib.sha256(payload).hexdigest() != manifest.get("corpus_sha256"):
        raise ValueError("Invalid corpus version or checksum")
    records = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if not records or len(records) != manifest["declaration_count"]:
        raise ValueError("Corpus declaration count mismatch")
    if len({r["id"] for r in records}) != len(records) or len({(r["module"], r["name"]) for r in records}) != len(records):
        raise ValueError("Duplicate declaration ID or module/name pair")
    for record in records:
        if record["commit"] != manifest["commit"] or record["repository"] != manifest["repository"]:
            raise ValueError("Corpus records come from different commits")
    return records, manifest
