import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stoogle.corpus import read_corpus, write_corpus
from stoogle.index import build_index
from stoogle.published import parse_page
from stoogle.search import Search
from stoogle.vector_cache import encode_cached
from stoogle.web_security import RequestLimitMiddleware

SAMPLE = Path(__file__).parent / "fixtures/parametric-family"
SHA = "a" * 40
ORIGIN = "https://github.com/StatLean/Stat-Lean"


def html(kind="noncomputable def", sha=SHA):
    return f'''<main><div class="decl" id="Example.foo"><div class="def">
    <div class="gh_link"><a href="{ORIGIN}/blob/{sha}/StatLean/Example.lean#L1-L1">source</a></div>
    <div class="decl_header"><span class="decl_kind">{kind}</span> <span class="decl_name">Example.foo</span>
    <span class="impl_arg">[MeasurableSpace Ω]</span><span class="decl_type">: a &lt; b</span></div>
    <p>The inequality <code>a&lt;b</code> is preserved.</p></div></div></main>'''


def test_compiled_html(monkeypatch):
    monkeypatch.setattr("stoogle.published.git", lambda *args: "noncomputable def foo := 1\n")
    records, omitted = parse_page(html(), "StatLean.Example", "StatLean/Example.lean", Path("."), SHA, ORIGIN, {})
    assert records[0]["kind"] == "definition"
    assert "[MeasurableSpace Ω]" in records[0]["signature"]
    assert "a<b" in records[0]["docstring"]
    assert records[0]["proof_status"] == "not_audited"
    assert not omitted


def test_mixed_published_commit():
    with pytest.raises(ValueError, match="not built from commit"):
        parse_page(html(sha="b"*40), "StatLean.Example", "StatLean/Example.lean", Path("."), SHA, ORIGIN, {})


def test_unknown_kind_fails(monkeypatch):
    monkeypatch.setattr("stoogle.published.git", lambda *args: "def foo := 1\n")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_page(html("mystery"), "StatLean.Example", "StatLean/Example.lean", Path("."), SHA, ORIGIN, {})


def test_vector_reuse_and_model_invalidation(tmp_path):
    calls = []
    def encode(texts):
        calls.append(texts)
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)
    model = SimpleNamespace(encode=encode)
    _, stats = encode_cached(model, ["one", "two"], "model", SHA, tmp_path)
    assert stats == {"reused": 0, "encoded": 2}
    _, stats = encode_cached(model, ["one", "three"], "model", SHA, tmp_path)
    assert stats == {"reused": 1, "encoded": 1}
    _, stats = encode_cached(model, ["one"], "model", "b"*40, tmp_path)
    assert stats["encoded"] == 1
    assert calls == [["one", "two"], ["three"], ["one"]]


def test_same_name_different_modules(tmp_path):
    records, manifest = read_corpus(SAMPLE)
    first = dict(records[0])
    second = {**first, "id": "different_id", "module": "StatLean.Another"}
    for record in (first, second):
        record["name_conflict_modules"] = [first["module"], second["module"]]
    manifest["declaration_count"] = 2
    corpus, index = tmp_path / "corpus", tmp_path / "index"
    write_corpus(corpus, [first, second], manifest)
    build_index(corpus, index, mode="lexical")
    async def scenario():
        search = Search(index)
        try:
            result = await search.search(first["name"])
            assert len(result["results"]) == 2
            assert len({r["id"] for r in result["results"]}) == 2
            with pytest.raises(ValueError, match="multiple modules"):
                search.get(first["name"])
            assert search.get(second["id"])["module"] == "StatLean.Another"
        finally:
            await search.close()
    asyncio.run(scenario())


def test_rate_limit():
    app = FastAPI()
    app.add_middleware(RequestLimitMiddleware, per_minute=2)
    @app.get("/api/search")
    async def find():
        return {}
    with TestClient(app) as client:
        assert client.get("/api/search").status_code == 200
        assert client.get("/api/search").status_code == 200
        response = client.get("/api/search")
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"
