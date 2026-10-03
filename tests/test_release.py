import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

from stoogle.corpus import read_corpus, write_corpus
from stoogle.index import build_index
from stoogle.release import fetch_release, package_index, sha256_file
from stoogle.search import Search

SAMPLE = Path(__file__).parent / "fixtures/parametric-family"
NAME = "statlean-2026-08-31-855b6afb"


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    root = tmp_path_factory.mktemp("release")
    records, manifest = read_corpus(SAMPLE)
    write_corpus(root / "corpus", records, {**manifest, "commit_date": "2026-08-31", "module_count": 25})
    build_index(root / "corpus", root / "index", mode="lexical")
    result = package_index(root / "index", root / "dist")
    assert result["release"] == NAME
    return root / "dist"


def serve(monkeypatch, files, tag=NAME):
    """Answer GitHub release API and asset requests from local files."""
    assets = [{"name": name, "browser_download_url": f"https://assets.example/{name}"} for name in files]

    def handler(request):
        if request.url.host == "api.github.com":
            if not request.url.path.endswith(("/latest", f"/tags/{tag}")):
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"tag_name": tag, "assets": assets})
        return httpx.Response(200, content=files[request.url.path.lstrip("/")])

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def published(dist):
    return {p.name: p.read_bytes() for p in dist.iterdir() if p.name.endswith((".tar.gz", ".sha256"))}


def test_package_contents(dist):
    archive = dist / f"{NAME}.tar.gz"
    assert (dist / f"{NAME}.tar.gz.sha256").read_text().split()[0] == sha256_file(archive)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert f"{NAME}/manifest.json" in names and f"{NAME}/index.json" in names
    notes = (dist / f"{NAME}.md").read_text()
    assert "855b6afb69fead1bef066111732ed44df181040e" in notes and "in 25 modules" in notes


def test_fetch_and_search_default_index(dist, tmp_path, monkeypatch):
    serve(monkeypatch, published(dist))
    result = fetch_release(home=tmp_path)
    assert result["release"] == NAME and (tmp_path / "current").resolve() == tmp_path / NAME
    assert asyncio.run(Search(tmp_path / "current").search("score_mean_zero", limit=1))["results"]
    assert fetch_release(NAME, home=tmp_path)["index"] == str(tmp_path / NAME)  # already present
    with pytest.raises(ValueError, match="No release"):
        fetch_release("statlean-missing", home=tmp_path)
    proc = subprocess.run([sys.executable, "-m", "stoogle.cli", "search", "score_mean_zero", "--limit", "1"],
                          capture_output=True, text=True, timeout=90, env={**os.environ, "STOOGLE_HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["results"][0]["name"] == "AsymptoticStatistics.Score.score_mean_zero"


def test_fetch_rejects_bad_checksum(dist, tmp_path, monkeypatch):
    files = published(dist)
    files[f"{NAME}.tar.gz.sha256"] = b"0" * 64 + b"\n"
    serve(monkeypatch, files)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_release(home=tmp_path)
    assert not (tmp_path / NAME).exists() and not (tmp_path / "current").exists()


def test_fetch_rejects_unsafe_archive(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    data = buffer.getvalue()
    serve(monkeypatch, {f"{NAME}.tar.gz": data,
                        f"{NAME}.tar.gz.sha256": hashlib.sha256(data).hexdigest().encode()})
    with pytest.raises(ValueError, match="Unsafe path"):
        fetch_release(home=tmp_path / "home")
    assert not (tmp_path / "escape.txt").exists()


def test_missing_default_index(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "stoogle.cli", "search", "score"], capture_output=True, text=True,
                          timeout=60, env={**os.environ, "STOOGLE_HOME": str(tmp_path / "empty")})
    assert proc.returncode == 2 and "stoogle fetch" in proc.stderr
