import asyncio
import json
from pathlib import Path

import httpx
import pytest

from stoogle.corpus import read_corpus, write_corpus
from stoogle.describe import apply_descriptions, dependency_levels, generate_descriptions
from stoogle.index import build_index, search_text
from stoogle.search import Search

SAMPLE = Path(__file__).parent / "fixtures/parametric-family"


def small_corpus(tmp_path):
    records, manifest = read_corpus(SAMPLE)
    records = [dict(r) for r in records[:6]]
    records[1]["signature_dependencies"] = [records[0]["name"]]
    records[2]["signature_dependencies"] = [records[1]["name"]]
    records[2]["docstring"] = ""
    manifest["declaration_count"] = len(records)
    write_corpus(tmp_path / "corpus", records, manifest)
    return tmp_path / "corpus", records


def test_dependency_levels(tmp_path):
    _, records = small_corpus(tmp_path)
    levels, _ = dependency_levels(records)
    position = {i: w for w, level in enumerate(levels) for i in level}
    assert position[0] < position[1] < position[2]


def test_generate_and_apply(tmp_path, monkeypatch):
    corpus, records = small_corpus(tmp_path)
    prompts = []

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        prompts.append(prompt)
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think></think>**Score.** The score has mean zero."}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    out = tmp_path / "descriptions.json"
    result = asyncio.run(generate_descriptions(corpus, out, "http://llm/v1", "fake-model", scope="all", concurrency=2))
    assert result["descriptions"] == len(records) and len(prompts) == len(records)
    # A dependency's text is passed as context when describing its dependents.
    assert any(records[0]["name"] in p and f"'{records[1]['name']}'" in p for p in prompts)
    document = json.loads(out.read_text())
    assert document["entries"][0]["description"] == "**Score.** The score has mean zero."

    # A later snapshot reuses earlier descriptions for unchanged declarations.
    prompts.clear()
    reused = asyncio.run(generate_descriptions(corpus, tmp_path / "next.json", "http://llm/v1", "fake-model",
                                               scope="all", cache=tmp_path / "next.cache.jsonl", reuse=[out]))
    assert prompts == [] and reused["reused"] == len(records)
    with pytest.raises(ValueError, match="different model"):
        asyncio.run(generate_descriptions(corpus, tmp_path / "other.json", "http://llm/v1", "other-model",
                                          cache=tmp_path / "other.cache.jsonl", reuse=[out]))

    # Re-running resumes from the cache without new requests.
    prompts.clear()
    again = asyncio.run(generate_descriptions(corpus, tmp_path / "again.json", "http://llm/v1", "fake-model",
                                              scope="missing-docstring", cache=out.with_suffix(".cache.jsonl")))
    assert prompts == [] and again["descriptions"] >= 1

    applied = tmp_path / "applied"
    apply_descriptions(corpus, out, applied)
    described, manifest = read_corpus(applied)
    assert all(r["generated_description"].startswith("**Score.**") for r in described)
    assert manifest["generated_description_model"] == "fake-model"
    assert "mean zero" in search_text(described[0])

    index = tmp_path / "index"
    build_index(applied, index, mode="lexical")
    assert (index / "generated-descriptions.json").exists()
    async def scenario():
        search = Search(index)
        assert (await search.search("score has mean zero", limit=3))["results"]
    asyncio.run(scenario())


def test_apply_rejects_stale_signature(tmp_path):
    corpus, records = small_corpus(tmp_path)
    _, manifest = read_corpus(corpus)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"commit": manifest["commit"], "model": "m", "entries": [
        {"id": records[0]["id"], "name": records[0]["name"], "signature_sha256": "0" * 64, "description": "x"}]}))
    with pytest.raises(ValueError, match="signature mismatch"):
        apply_descriptions(corpus, stale, tmp_path / "out")
    assert not (tmp_path / "out").exists()
