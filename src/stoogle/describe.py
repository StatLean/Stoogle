"""LLM-generated informal descriptions for retrieval, following LeanExplore.

Declarations are described in signature-dependency order, so the prompt for a
lemma can include descriptions of the concepts its statement mentions. Any
OpenAI-compatible chat endpoint works (a local vLLM server or a hosted API).
Descriptions are stored apart from the corpus, keyed by declaration ID and
signature hash.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from stoogle.corpus import read_corpus, write_corpus

# Adapted from LeanExplore's extract/prompt.txt (Apache-2.0); see NOTICE.
PROMPT = """**Goal:** Translate the formal Lean 4 declaration below into a clear, concise, purely informal English description, as it would appear in a statistics or probability textbook. The description will be used to find this declaration from a natural-language search query.

* **Reference material:** Base the description on the formal statement and the docstring. The prerequisite context explains concepts the statement mentions; use it for understanding, not as a source of claims.
* **Focus:** State what is defined or asserted: the objects involved, the essential hypotheses, and the conclusion. Be mathematically faithful, not merely a topic summary. Do not describe the proof.
* **Informal naming:** If the result or concept has a standard name (e.g. "Cramér–Rao inequality", "differentiability in quadratic mean", "Hoeffding's inequality"), use it. Mention the statistical setting when it is clear.
* **No Lean syntax:** Do not repeat the declaration name ('{name}'), code identifiers, type-class names, universe or type variables, keywords or tactics. Rephrase everything in mathematical English; use LaTeX with `$` delimiters for formulas.
* **Length:** One to three sentences.
* **Output format:** exactly
  **<informal name>.** <informal description>
  with no preamble or commentary.

**Declaration name:** {name}
**Kind:** {kind}

**Docstring:**
{docstring}

**Formal statement and source:**
```lean
{source}
```

**Context from prerequisites (for understanding only):**
{dependencies}

**Informal English description:**"""
MAX_SOURCE_CHARS = 4000
MAX_DOCSTRING_CHARS = 3000
MAX_DEPENDENCIES = 20
MAX_DEPENDENCY_CHARS = 256


def signature_sha256(record):
    return hashlib.sha256(record["signature"].encode()).hexdigest()


def needs_description(record, scope):
    if scope == "all":
        return True
    if scope == "missing-docstring":
        return not record["docstring"].strip()
    raise ValueError("Scope must be 'all' or 'missing-docstring'")


def dependency_levels(records):
    """Group record indices so every in-corpus signature dependency comes in an earlier level."""
    names = Counter(r["name"] for r in records)
    by_name = {r["name"]: i for i, r in enumerate(records) if names[r["name"]] == 1}
    deps = [[by_name[d] for d in r["signature_dependencies"] if d in by_name and by_name[d] != i]
            for i, r in enumerate(records)]
    depth = {}

    def visit(i, stack):
        if i in depth:
            return depth[i]
        if i in stack:  # mutual references: break the cycle
            return 0
        stack.add(i)
        depth[i] = 1 + max((visit(j, stack) for j in deps[i]), default=-1)
        stack.discard(i)
        return depth[i]

    for i in range(len(records)):
        visit(i, set())
    levels = [[] for _ in range(max(depth.values(), default=0) + 1)]
    for i, d in depth.items():
        levels[d].append(i)
    return levels, deps


def clean_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return " ".join(text.split())


async def generate_descriptions(corpus: Path, output: Path, endpoint: str, model: str,
                                scope="missing-docstring", cache: Path | None = None,
                                concurrency=32, api_key: str | None = None, temperature=0.3,
                                reuse: list[Path] = ()):
    """`reuse` takes description files from earlier snapshots; entries whose
    declaration ID and signature are unchanged are kept instead of regenerated."""
    import httpx

    records, manifest = read_corpus(corpus)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    cache_path = cache or output.with_suffix(".cache.jsonl")
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["model"] == model and entry["prompt_sha256"] == prompt_sha:
                done[(entry["id"], entry["signature_sha256"])] = entry["description"]
    for path in reuse:
        previous = json.loads(Path(path).read_text(encoding="utf-8"))
        if previous["model"] != model or previous["prompt_sha256"] != prompt_sha:
            raise ValueError(f"{path} was generated with a different model or prompt")
        for entry in previous["entries"]:
            done.setdefault((entry["id"], entry["signature_sha256"]), entry["description"])
    levels, deps = dependency_levels(records)
    generated, failures = {}, {}
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def context(i):
        lines = []
        for j in deps[i][:MAX_DEPENDENCIES]:
            dep = records[j]
            text = generated.get(j) or dep["description"] or dep["docstring"]
            if text:
                text = " ".join(text.split())
                if len(text) > MAX_DEPENDENCY_CHARS:
                    text = text[:MAX_DEPENDENCY_CHARS - 3] + "..."
                lines.append(f"- {dep['name']}: {text}")
        return "\n".join(lines) or "None"

    async def describe(client, i, sink):
        record = records[i]
        key = (record["id"], signature_sha256(record))
        if key in done:
            generated[i] = done[key]
            return
        prompt = PROMPT.format(name=record["name"], kind=record["kind"],
                               docstring=record["docstring"].strip()[:MAX_DOCSTRING_CHARS] or "No docstring available",
                               source=record["source_text"][:MAX_SOURCE_CHARS], dependencies=context(i))
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": 400,
                # Qwen3-style models: answer directly instead of emitting a reasoning trace.
                "chat_template_kwargs": {"enable_thinking": False}}
        async with semaphore:
            for attempt in range(4):
                try:
                    response = await client.post(endpoint.rstrip("/") + "/chat/completions", json=body,
                                                 headers=headers, timeout=300)
                    response.raise_for_status()
                    text = clean_output(response.json()["choices"][0]["message"]["content"] or "")
                    if not text:
                        raise ValueError("empty completion")
                    break
                except httpx.HTTPStatusError as error:
                    # A rejected request (e.g. over the context length) will not succeed on retry.
                    if error.response.status_code < 500 or attempt == 3:
                        failures[record["name"]] = f"HTTP {error.response.status_code}"
                        return
                    await asyncio.sleep(2 ** attempt)
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    if attempt == 3:
                        failures[record["name"]] = repr(error)
                        return
                    await asyncio.sleep(2 ** attempt)
        generated[i] = text
        sink.write(json.dumps({"id": record["id"], "signature_sha256": key[1], "model": model,
                               "prompt_sha256": prompt_sha, "description": text}, ensure_ascii=False) + "\n")
        sink.flush()

    async with httpx.AsyncClient() as client:
        with cache_path.open("a", encoding="utf-8") as sink:
            for level in levels:
                # Out-of-scope dependencies are context only; they need no generated text.
                await asyncio.gather(*(describe(client, i, sink) for i in level
                                       if needs_description(records[i], scope)))
    entries = [{"id": records[i]["id"], "name": records[i]["name"],
                "signature_sha256": signature_sha256(records[i]), "description": text}
               for i, text in sorted(generated.items())]
    document = {"commit": manifest["commit"], "corpus_sha256": manifest["corpus_sha256"],
                "model": model, "prompt_sha256": prompt_sha, "temperature": temperature,
                "scope": scope,
                "failures": failures, "entries": entries}
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"output": str(output), "descriptions": len(entries), "failures": len(failures),
            "reused": sum(1 for i in generated if (records[i]["id"], signature_sha256(records[i])) in done)}


def apply_descriptions(corpus: Path, descriptions: Path, output: Path):
    records, manifest = read_corpus(corpus)
    document = json.loads(descriptions.read_text(encoding="utf-8"))
    if document["commit"] != manifest["commit"]:
        raise ValueError("Descriptions belong to a different source commit")
    by_id = {r["id"]: r for r in records}
    for entry in document["entries"]:
        record = by_id.get(entry["id"])
        if record is None or signature_sha256(record) != entry["signature_sha256"]:
            raise ValueError(f"Description signature mismatch: {entry['name']}")
        record["generated_description"] = entry["description"]
    manifest.update({"generated_description_count": len(document["entries"]),
                     "generated_description_model": document["model"],
                     "generated_descriptions_sha256": hashlib.sha256(descriptions.read_bytes()).hexdigest()})
    write_corpus(output, records, manifest)
    for filename in ("coverage.json",):
        if (corpus / filename).exists():
            shutil.copyfile(corpus / filename, output / filename)
    shutil.copyfile(descriptions, output / "generated-descriptions.json")
    return {"declarations": len(records), "descriptions": len(document["entries"]), "commit": manifest["commit"]}
