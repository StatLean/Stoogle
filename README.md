# Stoogle

Natural-language search for [StatLean](https://github.com/StatLean/Stat-Lean), a Lean 4 library of formalized statistics. Describe a result in English or Chinese, or give its Lean name, and Stoogle returns the matching declarations with their signatures, the `import` they need and a link to their source. It indexes every public StatLean declaration, including intermediate lemmas that are not featured on the StatLean website, and can be used from the command line, a web page, an HTTP API or an MCP server.

## Quick start

Stoogle requires Python 3.10 or later.

```bash
git clone https://github.com/StatLean/Stoogle && cd Stoogle
python -m venv .venv && source .venv/bin/activate
pip install -e '.[semantic]'

stoogle fetch                                        # download the latest index
stoogle search --format agent 'the score function has mean zero'
stoogle serve                                        # web page at http://127.0.0.1:8000
```

## Usage

### Search

`stoogle search` prints JSON by default; `--format agent` prints each result's name, module, `import`, signature, source link and relevance as plain text. `--module` restricts results to a module prefix and `--kind` to `theorem`, `definition` or `structure`.

`stoogle serve` starts the web page and an HTTP API: `GET /api/search?q=...`, `GET /api/declarations/{id-or-name}` and `GET /api/info`, documented at `/docs`.

### MCP

`stoogle mcp` runs a read-only MCP server over stdio with three tools: `search_theorems`, `get_declaration` and `corpus_info`. To add it to an MCP client:

```json
{
  "mcpServers": {
    "stoogle": { "command": "/path/to/Stoogle/.venv/bin/stoogle", "args": ["mcp"] }
  }
}
```

`stoogle mcp --transport streamable-http --port 8001` serves it over HTTP instead. Both servers listen on the loopback interface by default.

### Indexes

Each index covers one StatLean commit and is published as a GitHub Release of this repository named `statlean-<commit date>-<commit>`, for example `statlean-2026-10-11-d41b0bb4`. The release notes give the full commit, the Lean and Mathlib versions and the models used.

`stoogle fetch` downloads the latest release, verifies its checksum and makes it the default index; `--release` selects another release and `--list` lists them. Indexes are stored in `$STOOGLE_HOME` (default `~/.cache/stoogle`), and `--index` points any command at a different index directory. The embedding and reranking models are downloaded from Hugging Face on first use.

## How it works

Stoogle follows the retrieval design of [LeanExplore](https://github.com/justincasher/lean-explore), which searches Mathlib.

1. **Descriptions.** When an index is built, an LLM writes a short textbook-style description of each declaration. Declarations are described in dependency order, so the description of a lemma can draw on those of the concepts it uses.
2. **Retrieval.** Keyword search (BM25) over names and over descriptions and signatures is combined with semantic search using [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
3. **Reranking.** [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) scores the top 20 candidates against the query, and they are reordered by their reranker rank combined with their retrieval rank. Its score is reported as `relevance`; when no candidate reaches 0.9, the response sets `low_confidence`, meaning StatLean may not contain the result.

Exact Lean names always rank first. Reranking is on by default when a GPU is available and takes over ten seconds per query on a CPU; `--rerank on|off` overrides the default.

## Building an index

```bash
# Import a StatLean commit from its published documentation.
stoogle import-published --repo /path/to/Stat-Lean --commit <commit SHA> \
  --cache build/docs-cache --output build/corpus

# Write descriptions with any OpenAI-compatible endpoint, e.g. `vllm serve Qwen/Qwen3-14B --port 8100`.
stoogle describe --corpus build/corpus --output build/descriptions.json --scope all \
  --endpoint http://127.0.0.1:8100/v1 --model Qwen/Qwen3-14B
stoogle apply-descriptions --corpus build/corpus --descriptions build/descriptions.json \
  --output build/corpus-described

stoogle build --corpus build/corpus-described --output build/index --embedding-cache build/vector-cache
```

- `import-published` reads the module list from the StatLean repository and stops if any module's documentation is missing or was built from another commit. The StatLean website documents only its latest commit; for an older commit, build the documentation with doc-gen4 and import its SQLite database with `stoogle import-sqlite`.
- `describe` can be resumed after an interruption, and `--reuse <descriptions.json>` keeps the descriptions of declarations whose signature has not changed since an earlier index. Describing all of StatLean with Qwen3-14B takes about 50 minutes on an A6000; a hosted endpoint's API key is read from `STOOGLE_LLM_API_KEY`.
- The description step is optional: `build` also accepts the imported corpus. `build --mode lexical` builds a keyword-only index, which needs neither PyTorch (`pip install -e .`) nor model downloads.

To publish an index, check it on a query set and package it:

```bash
stoogle evaluate --index build/index --queries queries.json
stoogle package --index build/index --output-dir dist
```

A query set is a JSON list of `{"query": ..., "relevant": [<Lean names>]}` objects; `stoogle heldout` creates one from the informal statements on the StatLean website.

`package` writes the archive, its SHA-256 checksum and release notes. Publish the archive and checksum as a GitHub Release named after the archive, with the notes as its description.

## License

Stoogle is released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions.
