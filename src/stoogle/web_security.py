"""Per-process request budget for the search and MCP endpoints."""
import os
import time
from collections import deque

from starlette.responses import JSONResponse


class RequestLimitMiddleware:
    def __init__(self, app, per_minute=None):
        self.app = app
        self.limit = int(per_minute or os.environ.get("STOOGLE_REQUESTS_PER_MINUTE", "120"))
        self.requests = deque()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and (scope["path"].startswith("/api/search") or scope["path"].startswith("/mcp")):
            now = time.monotonic()
            while self.requests and self.requests[0] < now - 60:
                self.requests.popleft()
            if len(self.requests) >= self.limit:
                await JSONResponse({"detail": "Server search budget exceeded; retry later"}, status_code=429,
                                   headers={"Retry-After": "60"})(scope, receive, send)
                return
            self.requests.append(now)
        await self.app(scope, receive, send)
