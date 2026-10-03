"""Build a reproducible Stoogle index from a corpus."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from stoogle.corpus import read_corpus

SCHEMA_VERSION = 2
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-0.6B"
# Instruction-aware embedders (Qwen3, e5-instruct) encode queries with a task
# prefix and documents without one; symmetric models use neither.
QUERY_INSTRUCTION = ("Instruct: Given a description of a statistics theorem or definition, "
                     "retrieve the matching Lean declaration\nQuery: ")
MAX_SEQ_LENGTH = 512


def query_prefix_for(model: str) -> str:
    return QUERY_INSTRUCTION if re.search(r"Qwen3-Embedding|e5-.*instruct", model) else ""


# Tokenizers follow LeanExplore's tokenization.py (Apache-2.0); see NOTICE.
def tokenize_spaced(text: str) -> list[str]:
    if not text:
        return []
    text = text.replace(".", " ").replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"\w+", text.lower())


def tokenize_raw(text: str) -> list[str]:
    return [text.lower()] if text else []


def search_text(record):
    # Human prose precedes the signature so long types do not crowd it out.
    return "\n".join(filter(None, [
        record["title"], record["description"], record.get("generated_description", ""), record["citation"],
        " ".join(tokenize_spaced(record["name"].split(".")[-1])),
        record["signature"],
    ]))


def compact_signature(signature):
    # Implicit and instance binders rarely carry meaning a reader searches for.
    signature = re.sub(r"\n\s*\{[^{}]*\}", "", signature)
    signature = re.sub(r"\n\s*\[[^\[\]]*\]", "", signature)
    return signature


def rerank_text(record):
    """Document shown to the cross-encoder: what a reader would judge a match by."""
    short = " ".join(tokenize_spaced(record["name"].split(".")[-1]))
    module = " ".join(tokenize_spaced(record["module"].removeprefix("StatLean.")))
    return "\n".join(filter(None, [
        f"{short} ({record['kind']}, {module})", record["title"], record["description"], record.get("generated_description", ""),
        compact_signature(record["signature"]),
    ]))


def device_for(device=None):
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embeddings:
    """SentenceTransformer wrapper with the query/document asymmetry recorded in the index."""
    def __init__(self, model: str, revision: str | None, query_prefix: str = "", device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
        transformers_logging.disable_progress_bar()
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.device = device_for(device)
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if self.device == "cuda" else {}
        self.model = SentenceTransformer(model, revision=revision, device=self.device,
                                         trust_remote_code=False, **kwargs)
        if (self.model.max_seq_length or 0) > MAX_SEQ_LENGTH:
            self.model.max_seq_length = MAX_SEQ_LENGTH
        self.query_prefix = query_prefix

    def encode(self, texts):
        vectors = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        # Normalize in float32: fp16 inference leaves norms ~1e-3 away from 1.
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def encode_queries(self, texts):
        return self.encode([self.query_prefix + t for t in texts])


def pinned_revision(model, revision):
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if Path(model).exists():
        raise ValueError("Use a Hugging Face model ID so the model revision can be pinned")
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision).sha


def build_index(corpus: Path, output: Path, mode="hybrid", model=DEFAULT_MODEL, revision=None,
                embedding_cache: Path | None = None, reranker: str | None = "auto",
                reranker_revision=None, device=None):
    """`reranker="auto"` records the default reranker for hybrid indexes only;
    lexical builds download no models."""
    import bm25s

    records, manifest = read_corpus(corpus)
    if output.exists():
        raise ValueError(f"Index already exists: {output}; build a new directory")
    if mode not in ("lexical", "hybrid"):
        raise ValueError("Mode must be lexical or hybrid")
    # Resolve model revisions before writing an index, and record their exact SHAs.
    embedder = None
    if mode == "hybrid":
        revision = pinned_revision(model, revision)
        embedder = Embeddings(model, revision, query_prefix_for(model), device)
    if reranker == "auto":
        reranker = DEFAULT_RERANKER if mode == "hybrid" else None
    if reranker:
        reranker_revision = pinned_revision(reranker, reranker_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stoogle-index-", dir=output.parent))
    try:
        for filename in ("corpus.jsonl", "manifest.json", "coverage.json", "generated-descriptions.json"):
            if (corpus / filename).exists():
                shutil.copyfile(corpus / filename, staging / filename)
        names = [r["name"] for r in records]
        for dirname, tokens in [
            ("bm25_name_raw", [tokenize_raw(n) for n in names]),
            ("bm25_name_spaced", [tokenize_spaced(n) for n in names]),
            ("bm25_text", [tokenize_spaced(search_text(r)) for r in records]),
        ]:
            bm25 = bm25s.BM25(method="lucene")
            bm25.index(tokens, show_progress=False)
            bm25.save(str(staging / dirname))
        embedding_stats = None
        if embedder is not None:
            from stoogle.vector_cache import encode_cached
            # The cache key includes the truncation length, which changes vectors.
            vectors, embedding_stats = encode_cached(
                embedder, [search_text(r) for r in records],
                f"{model}@{embedder.model.max_seq_length}", revision, embedding_cache)
            np.save(staging / "embeddings.npy", vectors.astype(np.float16), allow_pickle=False)
        (staging / "index.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "mode": mode,
            "model": model if embedder else None, "model_revision": revision if embedder else None,
            "query_prefix": embedder.query_prefix if embedder else None,
            "max_seq_length": embedder.model.max_seq_length if embedder else None,
            "reranker": reranker or None, "reranker_revision": reranker_revision if reranker else None,
            "rerank_top": 20, "low_confidence_below": 0.5,
            "corpus_sha256": manifest["corpus_sha256"],
            "embedding_cache": embedding_stats,
            "ranking": ("weighted RRF(k=60): name BM25 (1) + prose/signature BM25 (1) + dense cosine (2); "
                        "optional cross-encoder rerank of the top candidates; exact names first"),
        }, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"index": str(output), "mode": mode, "declarations": len(records)}
