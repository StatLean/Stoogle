"""Stoogle command-line interface."""
import argparse
import asyncio
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from contextlib import redirect_stdout
from pathlib import Path


RERANK = {"auto": None, "on": True, "off": False}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def render_agent_results(response, records, full_signatures=False):
    """Plain-text search results for agents and terminals."""
    snapshot = response["commit"][:12]
    if response.get("statlean_tag"):
        snapshot = f"{response['statlean_tag']} ({snapshot})"
    header = (
        f"Stoogle: {len(records)} candidate(s), StatLean {snapshot}, {response['mode']}, "
        f"{response['processing_time_ms']} ms"
    )
    lines = [header]
    if response["expanded_query"] != response["query"]:
        lines.append(f"Expanded query: {response['expanded_query']}")
    if not records:
        lines.append("No candidates found.")
        return "\n".join(lines)
    if response.get("low_confidence"):
        lines.append(f"Low confidence (best relevance {response['confidence']:.2f}): "
                     "StatLean may not contain this result.")

    for position, record in enumerate(records, 1):
        relevance = response["results"][position - 1].get("relevance")
        lines.extend([
            "",
            f"{position}. {record['name']} [{record['kind']}]"
            + (f"  relevance {relevance:.2f}" if relevance is not None else ""),
            f"   module: {record['module']}",
            f"   {record['import']}",
        ])
        description = " ".join(record.get("description", "").split())
        if description:
            lines.append(f"   summary: {description[:500]}{'…' if len(description) > 500 else ''}")
        elif record.get("generated_description"):
            generated = record["generated_description"]
            lines.append(f"   generated summary: {generated[:500]}{'…' if len(generated) > 500 else ''}")
        signature = record.get("signature", "").strip()
        if signature:
            if full_signatures:
                lines.append("   signature:")
                lines.extend(f"     {line}" for line in signature.splitlines())
            else:
                compact = " ".join(signature.split())
                lines.append(f"   signature: {compact[:700]}{'…' if len(compact) > 700 else ''}")
        lines.extend([
            f"   source: {record['source_link']}",
            f"   id: {record['id']}",
        ])
    return "\n".join(lines)


def index_path(args):
    from stoogle.release import default_index
    return args.index or default_index()


def open_search(args):
    from stoogle.search import Search
    return Search(index_path(args), rerank=RERANK[args.rerank], device=args.device)


def installed_versions():
    found = {}
    for package in ["stoogle", "bm25s", "numpy", "torch", "sentence-transformers", "transformers"]:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


async def query_index(args):
    search = open_search(args)
    try:
        response = await search.search(args.query, args.limit, args.module, args.kind)
        if args.format == "json":
            return response
        records = [search.by_id[result["id"]] for result in response["results"]]
        return render_agent_results(response, records, args.full_signatures)
    finally:
        await search.close()


async def evaluate(args):
    search = open_search(args)
    try:
        cases = json.loads(args.queries.read_text(encoding="utf-8"))
        if not cases:
            raise ValueError("Evaluation set is empty")
        rows = []
        for case in cases:
            gold = case["relevant"]
            if (not gold and not case.get("expect_absent")) or any(name not in search.by_name for name in gold):
                raise ValueError(f"Expected declaration not in the index for query: {case['query']}")
            result = await search.search(case["query"], limit=10)
            names = [r["name"] for r in result["results"]]
            rank = next((i for i, name in enumerate(names, 1) if name in gold), None)
            rows.append({"query": case["query"], "rank": rank, "top": names[:5],
                         "confidence": result["confidence"],
                         "expect_absent": bool(case.get("expect_absent")), "group": case.get("group", "development"),
                         "processing_time_ms": result["processing_time_ms"]})
        positives = [r for r in rows if not r["expect_absent"]]
        negatives = [r for r in rows if r["expect_absent"]]
        if not positives:
            raise ValueError("Evaluation needs at least one positive query")
        return {"mode": search.config["mode"], "reranked": search.rerank, "commit": search.manifest["commit"],
                "corpus_sha256": search.manifest["corpus_sha256"],
                "query_set_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                "model": search.config["model"], "model_revision": search.config["model_revision"],
                "reranker": search.config["reranker"] if search.rerank else None,
                "reranker_revision": search.config["reranker_revision"] if search.rerank else None,
                "versions": installed_versions(),
                "queries": len(positives), "total_queries": len(rows),
                "out_of_scope_queries": len(negatives), "out_of_scope_returned_candidates": sum(bool(r["top"]) for r in negatives),
                "out_of_scope_flagged_low_confidence": sum(r["confidence"] is not None and r["confidence"] < search.config.get("low_confidence_below", 0.5) for r in negatives),
                "hit_at_5": sum(r["rank"] is not None and r["rank"] <= 5 for r in positives) / len(positives),
                "mrr_at_10": sum(1 / r["rank"] if r["rank"] else 0 for r in positives) / len(positives), "results": rows}
    finally:
        await search.close()


def main():
    parser = argparse.ArgumentParser(prog="stoogle", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    published = commands.add_parser("import-published", help="Import a StatLean commit from its published doc-gen4 pages")
    published.add_argument("--repo", type=Path, required=True)
    published.add_argument("--commit", help="Full 40-character commit SHA")
    published.add_argument("--tag", help="StatLean tag; resolved to its commit and recorded in the manifest")
    published.add_argument("--base", default="https://statlean.github.io/docs")
    published.add_argument("--cache", type=Path, required=True)
    published.add_argument("--output", type=Path, required=True)
    sqlite = commands.add_parser("import-sqlite", help="Import declarations from a doc-gen4 SQLite database")
    sqlite.add_argument("--repo", type=Path, required=True)
    sqlite.add_argument("--database", type=Path, required=True)
    sqlite.add_argument("--prefix", action="append", required=True, help="Module prefix to include, e.g. StatLean")
    sqlite.add_argument("--output", type=Path, required=True)
    sqlite.add_argument("--tag", help="StatLean tag that must point to the database's commit")
    build = commands.add_parser("build", help="Build a search index from a corpus")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--mode", choices=["lexical", "hybrid"], default="hybrid")
    from stoogle.index import DEFAULT_MODEL, DEFAULT_RERANKER
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--revision")
    build.add_argument("--reranker", default="auto",
                       help=f"Hugging Face ID, or 'none' (default: {DEFAULT_RERANKER} for hybrid indexes)")
    build.add_argument("--reranker-revision")
    build.add_argument("--embedding-cache", type=Path)
    build.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    describe = commands.add_parser("describe", help="Generate informal descriptions with an OpenAI-compatible LLM endpoint")
    describe.add_argument("--corpus", type=Path, required=True)
    describe.add_argument("--output", type=Path, required=True)
    describe.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8100/v1")
    describe.add_argument("--model", required=True)
    describe.add_argument("--scope", choices=["missing-docstring", "all"], default="missing-docstring")
    describe.add_argument("--cache", type=Path, help="Resumable JSONL cache (default: next to --output)")
    describe.add_argument("--concurrency", type=int, default=32)
    describe.add_argument("--reuse", type=Path, action="append", default=[],
                          help="Descriptions from an earlier snapshot; unchanged declarations are not regenerated")
    apply = commands.add_parser("apply-descriptions", help="Add generated descriptions to a corpus (signature-checked)")
    apply.add_argument("--corpus", type=Path, required=True)
    apply.add_argument("--descriptions", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    heldout = commands.add_parser("heldout", help="Create a query set from the informal statements on the StatLean website")
    heldout.add_argument("--repo", type=Path, required=True)
    heldout.add_argument("--corpus", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)
    heldout.add_argument("--count", type=int, default=200)
    heldout.add_argument("--seed", type=int, default=0)
    package = commands.add_parser("package", help="Write a release archive, checksum and notes for an index")
    package.add_argument("--index", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, default=Path("dist"))
    package.add_argument("--name", help="Release name (default: statlean-<commit date>-<commit>)")
    package.add_argument("--report", type=Path, action="append", default=[],
                         help="Evaluation report to summarize in the release notes")
    fetch = commands.add_parser("fetch", help="Download a published index and make it the default")
    fetch.add_argument("--release", default="latest")
    fetch.add_argument("--dest", type=Path, help="Download directory (default: $STOOGLE_HOME or ~/.cache/stoogle)")
    fetch.add_argument("--repository", default="StatLean/Stoogle")
    fetch.add_argument("--list", action="store_true", help="List published releases instead")
    descriptions = {"search": "Search an index", "serve": "Serve the web page and HTTP API",
                    "mcp": "Run the MCP server", "evaluate": "Evaluate an index on a query set",
                    "info": "Show an index's StatLean version, models and settings"}
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--index", type=Path, help="Index directory (default: the last fetched release)")
        if name != "info":
            sub.add_argument("--rerank", choices=list(RERANK), default="auto",
                             help="Cross-encoder reranking (auto: only when a GPU is available)")
            sub.add_argument("--device", help="cuda or cpu (default: cuda when available)")
        if name == "search":
            sub.add_argument("query")
            sub.add_argument("--limit", type=int, default=10)
            sub.add_argument("--module")
            sub.add_argument("--kind")
            sub.add_argument("--format", choices=["json", "agent"], default="json")
            sub.add_argument("--full-signatures", action="store_true",
                             help="Do not compact signatures in agent output")
        if name in ("serve", "mcp"):
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8000 if name == "serve" else 8001)
        if name == "mcp":
            sub.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
        if name == "evaluate":
            sub.add_argument("--queries", type=Path, required=True)
            sub.add_argument("--output", type=Path, help="Write the report to this new JSON file")
    args = parser.parse_args()
    try:
        # Dependencies can print progress. Reserve stdout for JSON / MCP protocol.
        with redirect_stdout(sys.stderr):
            result = None
            if args.command == "import-published":
                from stoogle.published import export_published
                result = export_published(args.repo, args.commit, args.base, args.cache, args.output,
                                          tag=args.tag)
            elif args.command == "import-sqlite":
                from stoogle.corpus import export_corpus
                result = export_corpus(args.repo, args.database, args.prefix, args.output, args.tag)
            elif args.command == "build":
                from stoogle.index import build_index
                reranker = None if args.reranker.lower() == "none" else args.reranker
                result = build_index(args.corpus, args.output, args.mode, args.model, args.revision,
                                     args.embedding_cache, reranker, args.reranker_revision, args.device)
            elif args.command == "describe":
                import os
                from stoogle.describe import generate_descriptions
                result = asyncio.run(generate_descriptions(
                    args.corpus, args.output, args.endpoint, args.model, args.scope, args.cache,
                    args.concurrency, os.environ.get("STOOGLE_LLM_API_KEY"), reuse=args.reuse))
            elif args.command == "apply-descriptions":
                from stoogle.describe import apply_descriptions
                result = apply_descriptions(args.corpus, args.descriptions, args.output)
            elif args.command == "heldout":
                from stoogle.heldout import write_heldout
                result = write_heldout(args.repo, args.corpus, args.output, args.count, args.seed)
            elif args.command == "search":
                result = asyncio.run(query_index(args))
            elif args.command == "evaluate":
                result = asyncio.run(evaluate(args))
            elif args.command == "info":
                from stoogle.corpus import read_corpus
                index = index_path(args)
                _, result = read_corpus(index)
                result.update(json.loads((index / "index.json").read_text()))
            elif args.command == "package":
                from stoogle.release import package_index
                result = package_index(args.index, args.output_dir, args.name, args.report)
            elif args.command == "fetch":
                from stoogle.release import fetch_release, list_releases
                result = (list_releases(args.repository) if args.list
                          else fetch_release(args.release, args.dest, args.repository))
            elif args.command == "serve":
                import uvicorn
                from stoogle.api import create_app
                uvicorn.run(create_app(index_path(args), RERANK[args.rerank], args.device), host=args.host, port=args.port)
            elif args.command == "mcp":
                from stoogle.mcp import create_mcp
                server = create_mcp(index_path(args), args.host, args.port, RERANK[args.rerank], args.device)
        if args.command == "mcp":
            # The SDK captures stdout itself for stdio; model/index logs must not
            # corrupt that stream (BM25 and embeddings have progress disabled).
            if args.transport == "stdio":
                server.run(transport="stdio")
            else:
                import uvicorn
                from stoogle.web_security import RequestLimitMiddleware
                app = server.streamable_http_app()
                app.add_middleware(RequestLimitMiddleware)
                uvicorn.run(app, host=args.host, port=args.port)
        elif result is not None:
            if args.command == "evaluate" and args.output:
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            if isinstance(result, str):
                print(result)
            else:
                emit(result)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"stoogle: {exc}\n")


if __name__ == "__main__":
    main()
