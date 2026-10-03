"""Hybrid retrieval over a StatLean snapshot, with optional cross-encoder reranking.

The retrieval design follows LeanExplore (BM25 + dense embeddings fused by RRF,
then a Qwen3 reranker), implemented directly over a small in-memory corpus.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import bm25s
import numpy as np

from stoogle.corpus import read_corpus
from stoogle.index import SCHEMA_VERSION, Embeddings, device_for, rerank_text, tokenize_raw, tokenize_spaced

# Aliases expand only the keyword query; the dense and reranking channels read the original query.
ALIASES = {
    "dqm": "differentiability quadratic mean", "qmd": "differentiability quadratic mean",
    "lan": "local asymptotic normality", "clt": "central limit theorem",
    "得分函数": "score function", "费希尔信息": "fisher information",
    "平方均值可微": "differentiability quadratic mean", "二次均值可微": "differentiability quadratic mean",
    "期望为零": "expectation mean zero", "局部渐近正态": "local asymptotic normality",
    "中心极限定理": "central limit theorem", "核密度": "kernel density",
    "贝叶斯": "Bayesian", "伯恩斯坦冯米塞斯": "Bernstein von Mises",
    "逆概率加权": "inverse probability weighting", "假发现率": "false discovery rate",
    "中位数": "median", "经验风险": "empirical risk", "次高斯": "sub Gaussian",
}
RRF_K = 60
WEIGHTS = {"name": 1.0, "text": 1.0, "dense": 2.0}


def expand_query(query):
    extra = []
    tokens = tokenize_spaced(query)
    for alias, text in ALIASES.items():
        if (alias.isascii() and alias in tokens) or (not alias.isascii() and alias in query):
            extra.append(text)
    return " ".join([query, *extra])


class Reranker:
    """Qwen3-Reranker scored as P(yes) under its documented judgement prompt."""
    TASK = "Given a description of a statistics theorem or definition, judge whether the Lean declaration states it"
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
              "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model: str, revision: str, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
        transformers_logging.disable_progress_bar()
        self.torch = torch
        self.device = device_for(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=dtype).to(self.device).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    def score(self, query: str, documents: list[str], batch_size=10) -> np.ndarray:
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(documents), batch_size):
                prompts = [self.PREFIX + f"<Instruct>: {self.TASK}\n<Query>: {query}\n<Document>: {d}" + self.SUFFIX
                           for d in documents[start:start + batch_size]]
                batch = self.tokenizer(prompts, padding=True, truncation=True, max_length=1024,
                                       return_tensors="pt").to(self.device)
                logits = self.model(**batch).logits[:, -1, :]
                pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], 1).float()
                scores += pair.softmax(1)[:, 1].tolist()
        return np.asarray(scores)


class Search:
    def __init__(self, directory: Path, rerank: bool | None = None, device: str | None = None):
        """`rerank=None` reranks only on a GPU: on CPU a 0.6B cross-encoder over
        20 candidates takes tens of seconds per query."""
        directory = directory.resolve()
        self.records, self.manifest = read_corpus(directory)
        self.config = json.loads((directory / "index.json").read_text())
        if self.config.get("schema_version") != SCHEMA_VERSION or self.config.get("corpus_sha256") != self.manifest["corpus_sha256"]:
            raise ValueError("Index and corpus do not match; rebuild the index with this Stoogle version")
        if self.config.get("mode") not in ("lexical", "hybrid"):
            raise ValueError("Unknown index mode")
        if self.config["mode"] == "hybrid" and not (self.config.get("model") and self.config.get("model_revision")):
            raise ValueError("Hybrid index requires a pinned embedding model")
        self.by_id = {r["id"]: r for r in self.records}
        self.by_name = {r["name"]: r for r in self.records}
        self.ambiguous_names = {name for name, count in Counter(r["name"] for r in self.records).items() if count > 1}
        self.bm25 = {key: bm25s.BM25.load(str(directory / f"bm25_{key}"))
                     for key in ("name_raw", "name_spaced", "text")}
        self.vectors = None
        if self.config["mode"] == "hybrid":
            self.vectors = np.load(directory / "embeddings.npy", allow_pickle=False).astype(np.float32)
            if self.vectors.shape[0] != len(self.records):
                raise ValueError("Embedding matrix does not match the corpus")
        # Lexical indexes stay torch-free unless reranking is explicitly requested.
        if rerank is None:
            rerank = self.config["mode"] == "hybrid" and device_for(device) == "cuda"
        self.rerank = bool(self.config.get("reranker")) and rerank
        self.device = device
        self._embedder = None
        self._reranker = None
        self._rerank_texts = None
        # Models are lazily loaded and not re-entrant; queries are serialized.
        self.lock = asyncio.Lock()

    def info(self):
        return {**self.manifest, **self.config, "reranking": self.rerank}

    def get(self, identifier):
        if identifier in self.ambiguous_names:
            raise ValueError("This name occurs in multiple modules; search and use the declaration ID")
        return self.by_id.get(identifier) or self.by_name.get(identifier)

    async def close(self):
        pass

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embeddings(self.config["model"], self.config["model_revision"],
                                        self.config.get("query_prefix") or "", self.device)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker(self.config["reranker"], self.config["reranker_revision"], self.device)
            self._rerank_texts = [rerank_text(r) for r in self.records]
        return self._reranker

    def _bm25(self, key, tokens):
        if not tokens:
            return np.array([], dtype=int), np.array([])
        ids, scores = self.bm25[key].retrieve([tokens], k=len(self.records), show_progress=False)
        keep = scores[0] > 0  # positive lexical evidence only; OOV candidates do not vote
        return ids[0][keep].astype(int), scores[0][keep]

    def _fused(self, query, expanded):
        """Weighted RRF over all channels; returns candidate indices, best first."""
        n = len(self.records)
        name = np.zeros(n)
        for key, tokens in (("name_raw", tokenize_raw(expanded)), ("name_spaced", tokenize_spaced(expanded))):
            ids, scores = self._bm25(key, tokens)
            name[ids] = np.maximum(name[ids], scores)
        name_ids = np.flatnonzero(name)
        rankings = [("name", name_ids[np.lexsort((name_ids, -name[name_ids]))]),
                    ("text", self._bm25("text", tokenize_spaced(expanded))[0])]
        if self.vectors is not None:
            similarity = self.vectors @ self.embedder.encode_queries([query])[0]
            rankings.append(("dense", np.argsort(-similarity, kind="stable")))
        fused = np.zeros(n)
        voted = np.zeros(n, dtype=bool)
        for channel, order in rankings:
            fused[order] += WEIGHTS[channel] / (RRF_K + np.arange(1, len(order) + 1))
            voted[order] = True
        candidates = np.flatnonzero(voted)
        return candidates[np.lexsort((candidates, -fused[candidates]))].tolist()

    def _search(self, query, limit, module, kind):
        expanded = expand_query(query)
        order = self._fused(query, expanded)
        if module:
            order = [i for i in order if self.records[i]["module"] == module or self.records[i]["module"].startswith(module + ".")]
        if kind:
            order = [i for i in order if self.records[i]["kind"] == kind]
        relevance = {}
        if self.rerank and order:
            top = order[:max(limit, self.config.get("rerank_top", 20))]
            scores = self.reranker.score(query, [self._rerank_texts[i] for i in top])
            relevance = {i: float(s) for i, s in zip(top, scores)}
            top = [top[j] for j in np.argsort(-scores, kind="stable")]
            order = top + order[len(top):]
        folded = query.casefold()
        exact = [i for i in order if folded in (self.records[i]["name"].casefold(),
                                                self.records[i]["name"].rsplit(".", 1)[-1].casefold())]
        order = sorted(exact, key=lambda i: self.records[i]["name"]) + [i for i in order if i not in set(exact)]
        return expanded, order[:limit], relevance

    async def search(self, query: str, limit=10, module=None, kind=None):
        if not isinstance(query, str) or len(query) > 2000:
            raise ValueError("Query must be a string of at most 2000 characters")
        if not 1 <= limit <= 50:
            raise ValueError("Limit must be between 1 and 50")
        start = time.perf_counter()
        query = query.strip()
        expanded, chosen, relevance = expand_query(query), [], {}
        if query:
            async with self.lock:
                # Inference runs off the event loop so HTTP/MCP health checks stay responsive.
                expanded, chosen, relevance = await asyncio.to_thread(self._search, query, limit, module, kind)
        fields = ("id", "name", "title", "description", "module", "kind", "featured", "proof_status", "source_link")
        results = [{**{k: self.records[i][k] for k in fields},
                    "generated_description": self.records[i].get("generated_description"),
                    "relevance": relevance.get(i)} for i in chosen]
        confidence = max(relevance.values(), default=None) if relevance else None
        return {
            "query": query, "expanded_query": expanded, "mode": self.config["mode"],
            "reranked": bool(relevance), "commit": self.manifest["commit"],
            "statlean_tag": self.manifest.get("statlean_tag"),
            "confidence": confidence,
            "low_confidence": confidence is not None and confidence < self.config.get("low_confidence_below", 0.5),
            "results": results, "count": len(results),
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 1),
        }
