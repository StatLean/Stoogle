"""Read-only HTTP API and a small same-origin search website."""
from contextlib import asynccontextmanager
from pathlib import Path
import os

from importlib.metadata import version

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from stoogle.search import Search
from stoogle.web_security import RequestLimitMiddleware


def create_app(index: Path, rerank=None, device=None) -> FastAPI:
    search = Search(index, rerank=rerank, device=device)

    @asynccontextmanager
    async def lifespan(app):
        if os.environ.get("STOOGLE_WARMUP") == "1" and search.config["mode"] == "hybrid":
            await search.search("Fisher information", limit=1)
        yield
        await search.close()

    app = FastAPI(title="Stoogle", version=version("stoogle"), lifespan=lifespan)
    app.add_middleware(RequestLimitMiddleware)
    origins = [x.strip() for x in os.environ.get("STOOGLE_ALLOWED_ORIGINS", "").split(",") if x.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=[])

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path not in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'self'"
        return response
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(static / "index.html")

    @app.get("/api/info")
    async def info():
        return search.info()

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok", "commit": search.manifest["commit"], "declarations": len(search.records)}

    @app.get("/api/search")
    async def find(q: str = Query(max_length=2000), limit: int = Query(10, ge=1, le=50),
                   module: str | None = None, kind: str | None = None):
        return await search.search(q, limit, module, kind)

    @app.get("/api/declarations/{identifier:path}")
    async def declaration(identifier: str):
        try:
            record = search.get(identifier)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if record is None:
            raise HTTPException(404, "Declaration not in this snapshot")
        return record

    return app
