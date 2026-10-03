import asyncio
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stoogle.corpus import RenderedCode, export_corpus, plain, read_corpus, source_identity
from stoogle.index import build_index
from stoogle.search import Search, expand_query

SAMPLE = Path(__file__).parent / "fixtures" / "parametric-family"
SCORE = "AsymptoticStatistics.Score.score_mean_zero"


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    path = tmp_path_factory.mktemp("search") / "index"
    build_index(SAMPLE, path, mode="lexical")
    return path


def run(coro):
    return asyncio.run(coro)


def test_snapshot():
    records, manifest = read_corpus(SAMPLE)
    assert len(records) == 154
    assert manifest["commit"] == "855b6afb69fead1bef066111732ed44df181040e"
    assert manifest["repository"] == "https://github.com/StatLean/Stat-Lean"
    score = next(r for r in records if r["name"] == SCORE)
    assert score["import"] == "import StatLean.AsymptoticStatistics.ParametricFamily.Score"
    assert score["signature"].split()[:2] == ["theorem", SCORE]
    assert "[" in score["signature"] and "(hDQM" in score["signature"]
    assert "∫" in score["signature"]
    assert score["commit"] in score["source_link"]
    assert {r["proof_status"] for r in records} == {"not_audited"}
    assert any(r["signature_dependencies"] for r in records)


@pytest.mark.parametrize("url", ["https://github.com/a/b/blob/main/X.lean", "https://example.com/a/b/blob/" + "a"*40 + "/X.lean", "file:///tmp/X"])
def test_reject_unpinned_source(url):
    with pytest.raises(ValueError):
        source_identity(url)


def test_rendered_code():
    text = "∀ θ : ℝ, θ = θ"
    encoded = text.encode()
    assert RenderedCode(bytes([0, len(encoded)]) + encoded).decode() == (text, set())
    # tag(const(name(anonymous, "Real"))), text("ℝ")
    assert RenderedCode(b"\x01\x02\x01\x00\x04Real\x00\x03" + "ℝ".encode()).decode() == ("ℝ", {"Real"})


def test_markdown_inequalities_preserved():
    assert plain("Sum `∑_{i<n} Y i`\n with `a<b`.") == "Sum `∑_{i<n} Y i` with `a<b`."


@pytest.mark.parametrize("data", [b"", b"\x00\x09a", b"\x03", b"\x00\x00extra", b"\x01\x09"])
def test_bad_rendered_code(data):
    with pytest.raises(ValueError):
        RenderedCode(data).decode()


def test_checksum(tmp_path):
    copied = tmp_path / "corpus"
    shutil.copytree(SAMPLE, copied)
    with (copied / "corpus.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        read_corpus(copied)


def test_mismatched_index(index, tmp_path):
    copied = tmp_path / "index"
    shutil.copytree(index, copied)
    config = json.loads((copied / "index.json").read_text())
    config["corpus_sha256"] = "wrong"
    (copied / "index.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="do not match"):
        Search(copied)


@pytest.mark.skipif(not os.environ.get("STOOGLE_TEST_HYBRID_INDEX"), reason="set STOOGLE_TEST_HYBRID_INDEX to a hybrid index to run")
def test_real_hybrid():
    async def scenario():
        search = Search(Path(os.environ["STOOGLE_TEST_HYBRID_INDEX"]))
        try:
            assert search.config["mode"] == "hybrid"
            result = await search.search("得分函数的期望为零", limit=5)
            assert SCORE in [r["name"] for r in result["results"]]
            assert result["reranked"] and result["results"][0]["relevance"] is not None
            assert search.vectors.shape[0] == len(search.records)
            absent = await search.search("fastest route from Boston to Paris", limit=5)
            assert absent["low_confidence"]
        finally:
            await search.close()
    run(scenario())


def test_no_overwrite(index):
    with pytest.raises(ValueError, match="already exists"):
        build_index(SAMPLE, index, mode="lexical")


def test_query_expansion():
    assert "score function" in expand_query("得分函数的期望为零")
    assert "differentiability" in expand_query("DQM")
    assert expand_query("island") == "island"  # aliases match whole tokens, so "lan" does not fire


def test_search(index):
    async def scenario():
        search = Search(index)
        try:
            for query in (SCORE, "score_mean_zero"):
                result = await search.search(query, limit=5)
                assert result["results"][0]["name"] == SCORE
            chinese = await search.search("得分函数的期望为零", limit=5)
            assert SCORE in [r["name"] for r in chinese["results"]]
            assert (await search.search(""))["results"] == []
            assert (await search.search("zzzznoexistingtoken123"))["results"] == []
            assert (await search.search("!!!"))["results"] == []
            assert (await search.search("score", module="StatLean.Missing"))["results"] == []
            filtered = await search.search("score", module="StatLean.AsymptoticStatistics.DQM", kind="theorem")
            assert filtered["results"]
            assert all(r["module"].startswith("StatLean.AsymptoticStatistics.DQM.") and r["kind"] == "theorem" for r in filtered["results"])
            assert search.get("missing") is None
            assert search.get(search.get(SCORE)["id"])["name"] == SCORE
            for kwargs in ({"limit": 0}, {"limit": 51}):
                with pytest.raises(ValueError):
                    await search.search("score", **kwargs)
            with pytest.raises(ValueError):
                await search.search("x" * 2001)
            assert search._embedder is None and search._reranker is None  # keyword search loads no models
        finally:
            await search.close()
    run(scenario())


def test_concurrent_search(index):
    async def scenario():
        search = Search(index)
        try:
            results = await asyncio.gather(search.search("score_mean_zero"), search.search("fisherInformation"))
            assert results[0]["results"][0]["name"] == SCORE
            assert results[1]["results"][0]["name"] == "AsymptoticStatistics.fisherInformation"
        finally:
            await search.close()
    run(scenario())


def test_api(index):
    from stoogle.api import create_app
    with TestClient(create_app(index)) as client:
        assert client.get("/").status_code == 200
        assert "text/css" in client.get("/static/style.css").headers["content-type"]
        assert client.get("/api/info").json()["mode"] == "lexical"
        response = client.get("/api/search", params={"q": SCORE})
        assert response.status_code == 200
        key = response.json()["results"][0]["id"]
        assert client.get(f"/api/declarations/{key}").json()["name"] == SCORE
        assert client.get("/api/declarations/nope").status_code == 404
        assert client.get("/api/search", params={"q": "score", "limit": 51}).status_code == 422
        assert client.get("/api/search", params={"q": "x" * 2001}).status_code == 422


def test_cli_json(index):
    proc = subprocess.run([sys.executable, "-m", "stoogle.cli", "search", "--index", str(index), "score_mean_zero", "--limit", "1"], capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["results"][0]["name"] == SCORE


def test_cli_agent_output(index):
    proc = subprocess.run(
        [sys.executable, "-m", "stoogle.cli", "search", "--index", str(index),
         "score_mean_zero", "--limit", "1", "--format", "agent"],
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Stoogle: 1 candidate(s)" in proc.stdout
    assert SCORE in proc.stdout
    assert "import StatLean." in proc.stdout
    assert "signature:" in proc.stdout
    assert "source: https://github.com/StatLean/Stat-Lean/blob/" in proc.stdout


def test_real_mcp_stdio(index):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "stoogle.cli", "mcp", "--index", str(index)],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {t.name for t in tools} == {"search_theorems", "get_declaration", "corpus_info"}
                assert all(t.annotations.readOnlyHint for t in tools)
                result = await session.call_tool("search_theorems", {"query": "score_mean_zero", "limit": 1})
                assert not result.isError
                assert json.loads(result.content[0].text)["results"][0]["name"] == SCORE
                detail = await session.call_tool("get_declaration", {"identifier": SCORE})
                assert "import StatLean." in json.loads(detail.content[0].text)["import"]
                invalid = await session.call_tool("search_theorems", {"query": "score", "limit": 99})
                assert invalid.isError
    run(asyncio.wait_for(scenario(), timeout=90))


def test_real_mcp_http(index):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    proc = subprocess.Popen([sys.executable, "-m", "stoogle.cli", "mcp", "--index", str(index), "--transport", "streamable-http", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready = False
        for _ in range(100):
            assert proc.poll() is None, "MCP process exited early"
            try:
                ready = httpx.get(url, timeout=0.2).status_code in (200, 400, 405, 406)
            except httpx.HTTPError:
                pass
            if ready:
                break
            time.sleep(0.1)
        assert ready, "MCP HTTP readiness timeout"

        async def scenario():
            async with streamable_http_client(url) as (reader, writer, _):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    result = await session.call_tool("search_theorems", {"query": "score_mean_zero", "limit": 1})
                    assert not result.isError
                    assert json.loads(result.content[0].text)["results"][0]["name"] == SCORE
        run(asyncio.wait_for(scenario(), timeout=30))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_export_compiled_fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "StatLean").mkdir()
    source = "/- A comment with Unicode θ and a nested /- comment -/. -/\ntheorem example (θ : Nat) : θ = θ := by rfl\n"
    (repo / "StatLean" / "Example.lean").write_text(source)
    (repo / "lean-toolchain").write_text("leanprover/lean4:v4.29.1\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()
    git("init")
    git("add", ".")
    git("-c", "user.name=Stoogle Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    dbpath = tmp_path / "docs.db"
    def blob(text):
        data = text.encode()
        return bytes([0, len(data)]) + data
    with sqlite3.connect(dbpath) as db:
        db.executescript("""
            CREATE TABLE name_info(name, module_name, position, kind, type, sorried, render);
            CREATE TABLE modules(name, source_url);
            CREATE TABLE declaration_ranges(module_name, position, start_line, start_column, end_line, end_column);
            CREATE TABLE declaration_markdown_docstrings(module_name, position, text);
            CREATE TABLE declaration_args(module_name, position, sequence, binder);
        """)
        module = "StatLean.Example"
        db.execute("INSERT INTO name_info VALUES (?, ?, 0, 'theorem', ?, 1, 1)", ("example", module, blob("θ = θ")))
        db.execute("INSERT INTO modules VALUES (?, ?)", (module, f"https://github.com/example/statlean/blob/{commit}/StatLean/Example.lean"))
        db.execute("INSERT INTO declaration_ranges VALUES (?, 0, 2, 0, 2, ?)", (module, len(source.splitlines()[1])))
        db.execute("INSERT INTO declaration_args VALUES (?, 0, 0, ?)", (module, blob("(θ : Nat)")))
    output = tmp_path / "export"
    export_corpus(repo, dbpath, ["StatLean"], output)
    records, _ = read_corpus(output)
    assert records[0]["source_text"] == source.splitlines()[1]
    assert records[0]["signature"] == "theorem example\n    (θ : Nat) :\n    θ = θ"
    assert records[0]["proof_status"] == "contains_sorry"
    assert records[0]["description_origin"] == "none"


class FakeReranker:
    """Scores documents by whether they mention the query's first word."""
    def score(self, query, documents):
        word = query.split()[0].casefold()
        return np.asarray([0.9 if word in d.casefold() else 0.1 for d in documents])


def test_rerank_reorders_and_flags_low_confidence(index):
    async def scenario():
        search = Search(index)
        try:
            search.config["reranker"], search.config["reranker_revision"] = "fake", "0" * 40
            search.rerank, search._reranker = True, FakeReranker()
            search._rerank_texts = [r["name"] for r in search.records]
            result = await search.search("fisherInformation of the model", limit=5)
            assert result["reranked"] and not result["low_confidence"]
            assert "fisherinformation" in result["results"][0]["name"].casefold()
            assert result["results"][0]["relevance"] == 0.9
            missing = await search.search("zzzz score", limit=5)
            assert missing["low_confidence"] and missing["confidence"] == 0.1
            exact = await search.search("score_mean_zero", limit=3)
            assert exact["results"][0]["name"] == SCORE  # exact names stay pinned above reranking
        finally:
            await search.close()
    run(scenario())
