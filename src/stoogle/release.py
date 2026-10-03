"""Release archives of built indexes, published as GitHub Releases.

A release is named after the StatLean commit it indexes,
`statlean-<commit date>-<first 8 hex digits>`, and consists of a gzipped tar
of the index directory, its SHA-256 checksum and release notes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from stoogle.corpus import read_corpus

DEFAULT_REPOSITORY = "StatLean/Stoogle"
STATLEAN = "https://github.com/StatLean/Stat-Lean"
QUERY_SETS = {
    "curated": "Hand-written queries (English and Chinese)",
    "website": "Informal statements from the StatLean website",
    "intermediate": "Declarations not featured on the website",
}


def default_home() -> Path:
    if os.environ.get("STOOGLE_HOME"):
        return Path(os.environ["STOOGLE_HOME"])
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "stoogle"


def default_index() -> Path:
    """The index that `stoogle fetch` last activated."""
    current = default_home() / "current"
    if not current.exists():
        raise ValueError("No index given and none fetched; run `stoogle fetch` or pass --index")
    return current


def release_name(manifest: dict) -> str:
    if not manifest.get("commit_date"):
        raise ValueError("The corpus manifest has no commit_date; import it with this version of Stoogle")
    return f"statlean-{manifest['commit_date']}-{manifest['commit'][:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def release_notes(name: str, manifest: dict, config: dict, reports: list[Path]) -> str:
    commit = manifest["commit"]
    lines = [f"Search index for StatLean [`{commit[:8]}`]({STATLEAN}/tree/{commit}), committed {manifest['commit_date']}.", ""]
    if manifest.get("statlean_tag"):
        lines += [f"StatLean tag: `{manifest['statlean_tag']}`", ""]
    lean = manifest.get("lean_toolchain", "").split(":")[-1]
    mathlib = manifest.get("mathlib_input_rev") or ""
    rev = manifest.get("mathlib_rev", "")[:8]
    lines += [
        "| | |", "| --- | --- |",
        f"| StatLean commit | `{commit}` |",
        f"| Lean | {lean} |",
        f"| Mathlib | {mathlib} (`{rev}`) |" if rev else "| Mathlib | — |",
        f"| Declarations | {manifest['declaration_count']:,}"
        + (f" in {manifest['module_count']:,} modules |" if manifest.get("module_count") else " |"),
        f"| Index | {config['mode']} |",
    ]
    if config.get("model"):
        lines.append(f"| Embedding model | {config['model']} (`{config['model_revision'][:8]}`) |")
    if config.get("reranker"):
        lines.append(f"| Reranker | {config['reranker']} (`{config['reranker_revision'][:8]}`) |")
    if manifest.get("generated_description_model"):
        lines.append(f"| Descriptions | {manifest['generated_description_count']:,} by {manifest['generated_description_model']} |")
    if reports:
        lines += ["", "Hit@5, the share of queries whose expected declaration is among the top five results:",
                  "", "| Query set | Reranking | Hit@5 |", "| --- | --- | ---: |"]
        for path in reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("corpus_sha256") != manifest["corpus_sha256"]:
                raise ValueError(f"{path} evaluates a different corpus")
            hits = round(report["hit_at_5"] * report["queries"])
            label = QUERY_SETS.get(path.stem, path.stem)
            lines.append(f"| {label} | {'on' if report.get('reranked') else 'off'} | {hits}/{report['queries']} |")
    lines += ["", "```bash", f"stoogle fetch --release {name}", "```", ""]
    return "\n".join(lines)


def package_index(index: Path, output_dir: Path, name: str | None = None, reports: list[Path] = ()):
    _, manifest = read_corpus(index)
    config = json.loads((index / "index.json").read_text())
    if config.get("corpus_sha256") != manifest["corpus_sha256"]:
        raise ValueError("Index and corpus do not match")
    name = name or release_name(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{name}.tar.gz"
    if archive.exists():
        raise ValueError(f"Archive already exists: {archive}")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(index.resolve().rglob("*")):
            if path.is_file():
                tar.add(path, arcname=f"{name}/{path.relative_to(index.resolve()).as_posix()}", recursive=False)
    checksum = output_dir / f"{name}.tar.gz.sha256"
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n")
    notes = output_dir / f"{name}.md"
    notes.write_text(release_notes(name, manifest, config, list(reports)), encoding="utf-8")
    return {"release": name, "archive": str(archive), "checksum": str(checksum), "notes": str(notes),
            "bytes": archive.stat().st_size}


def _safe_extract(archive: Path, name: str, destination: Path):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or path.parts[0] != name
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe path in release archive: {member.name}")
        # The "data" filter (Python 3.12, backported to 3.10.12) also strips unsafe modes.
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(destination, members=members, **options)


def _point_current(home: Path, name: str):
    temporary = home / f".current-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(name, target_is_directory=True)
    os.replace(temporary, home / "current")


def list_releases(repository: str = DEFAULT_REPOSITORY):
    import httpx
    try:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get(f"https://api.github.com/repos/{repository}/releases",
                                  headers={"Accept": "application/vnd.github+json"})
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise ValueError(f"GitHub request failed: {error}") from error
    return [{"release": r["tag_name"], "published": r["published_at"]} for r in response.json()]


def fetch_release(release: str = "latest", home: Path | None = None, repository: str = DEFAULT_REPOSITORY):
    """Download, verify and activate a release; return the path of its index."""
    import httpx
    try:
        return _fetch(release, home, repository)
    except httpx.HTTPError as error:
        raise ValueError(f"Download from GitHub failed: {error}") from error


def _fetch(release, home, repository):
    import httpx
    home = home or default_home()
    home.mkdir(parents=True, exist_ok=True)
    path = "latest" if release == "latest" else f"tags/{release}"
    with httpx.Client(follow_redirects=True, timeout=600) as client:
        response = client.get(f"https://api.github.com/repos/{repository}/releases/{path}",
                              headers={"Accept": "application/vnd.github+json"})
        if response.status_code == 404:
            raise ValueError(f"No release {release!r} in {repository}")
        response.raise_for_status()
        info = response.json()
        name = info["tag_name"]
        assets = {asset["name"]: asset["browser_download_url"] for asset in info["assets"]}
        archive_name = f"{name}.tar.gz"
        if archive_name not in assets or archive_name + ".sha256" not in assets:
            raise ValueError(f"Release {name} has no index archive and checksum")
        target = home / name
        if not target.exists():
            expected = client.get(assets[archive_name + ".sha256"]).raise_for_status().text.split()[0]
            staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=home))
            try:
                archive = staging / archive_name
                with client.stream("GET", assets[archive_name]) as stream, archive.open("wb") as sink:
                    stream.raise_for_status()
                    for block in stream.iter_bytes():
                        sink.write(block)
                if sha256_file(archive) != expected:
                    raise ValueError(f"Checksum mismatch for {archive_name}")
                _safe_extract(archive, name, staging)
                read_corpus(staging / name)
                os.replace(staging / name, target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
    _, manifest = read_corpus(target)
    _point_current(home, name)
    return {"release": name, "index": str(target), "current": str(home / "current"),
            "statlean_commit": manifest["commit"]}
