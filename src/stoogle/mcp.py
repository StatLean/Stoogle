"""Official MCP SDK transport; all tools are read-only."""
from contextlib import asynccontextmanager
from pathlib import Path
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from mcp.server.transport_security import TransportSecuritySettings

from stoogle.search import Search


def create_mcp(index: Path, host="127.0.0.1", port=8001, rerank=None, device=None):
    search = Search(index, rerank=rerank, device=device)

    @asynccontextmanager
    async def lifespan(server):
        yield {}
        await search.close()

    server = FastMCP("stoogle", host=host, port=port, lifespan=lifespan,
                     transport_security=TransportSecuritySettings(
                         enable_dns_rebinding_protection=True,
                         allowed_hosts=["127.0.0.1:*", "localhost:*", *filter(None, os.environ.get("STOOGLE_MCP_HOSTS", "").split(","))],
                         allowed_origins=["http://127.0.0.1:*", "http://localhost:*", *filter(None, os.environ.get("STOOGLE_ALLOWED_ORIGINS", "").split(","))]),
                     instructions="Search StatLean declarations by natural language or Lean name.")
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=readonly)
    async def search_theorems(query: str, limit: int = 10,
                              module: str | None = None, kind: str | None = None) -> dict:
        """Find StatLean declarations by natural language (English or Chinese) or Lean name.

        Returns the snapshot commit. When reranked, each result has a relevance in [0, 1]
        and low_confidence flags queries the library may not cover.
        """
        return await search.search(query, limit, module, kind)

    @server.tool(annotations=readonly)
    async def get_declaration(identifier: str) -> dict:
        """Get a declaration by result ID or full Lean name: its signature, import, source and the constants its signature mentions."""
        record = search.get(identifier)
        if record is None:
            raise ValueError("Declaration not in this snapshot")
        return record

    @server.tool(annotations=readonly)
    async def corpus_info() -> dict:
        """Get corpus scope, exact source commit, index mode and pinned embedding model."""
        return search.info()

    return server
