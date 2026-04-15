#!/usr/bin/env python3
"""
Simple load balancer for multi-GPU llama.cpp servers

Forwards requests to the least loaded server instance.
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
from aiohttp import web
from collections import deque
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadBalancer:
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.active_requests: Dict[str, int] = {b: 0 for b in backends}
        self.request_counts: Dict[str, deque] = {b: deque(maxlen=100) for b in backends}

    def get_least_loaded(self) -> str:
        return min(self.backends, key=lambda b: self.active_requests[b])

    def increment(self, backend: str):
        self.active_requests[backend] += 1
        self.request_counts[backend].append(1)

    def decrement(self, backend: str):
        self.active_requests[backend] = max(0, self.active_requests[backend] - 1)

    def get_stats(self) -> dict:
        return {
            "backends": {
                b: {
                    "active_requests": self.active_requests[b],
                    "total_requests": sum(self.request_counts[b]),
                }
                for b in self.backends
            },
            "least_loaded": self.get_least_loaded(),
        }


class ProxyBackend:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.base_path = ""

    async def forward(self, request: web.Request, path: str) -> web.Response:
        import aiohttp

        url = f"{self.url}{path}"
        headers = dict(request.headers)
        headers.pop("Host", None)

        async with aiohttp.ClientSession() as session:
            try:
                if request.can_read_body:
                    body = await request.read()
                    async with session.request(
                        request.method, url, headers=headers, data=body, timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        content = await resp.read()
                        return web.Response(
                            body=content,
                            status=resp.status,
                            headers=dict(resp.headers),
                        )
                else:
                    async with session.request(
                        request.method, url, headers=headers, timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        content = await resp.read()
                        return web.Response(
                            body=content,
                            status=resp.status,
                            headers=dict(resp.headers),
                        )
            except Exception as e:
                logger.error(f"Error forwarding to {url}: {e}")
                return web.Response(status=502, text=f"Backend error: {str(e)}")


async def handle_request(request: web.Request, lb: LoadBalancer, path: str) -> web.Response:
    backend = lb.get_least_loaded()
    lb.increment(backend)

    proxy = ProxyBackend(backend)
    try:
        response = await proxy.forward(request, path)
        return response
    finally:
        lb.decrement(backend)


async def handle_health(request: web.Request, lb: LoadBalancer) -> web.Response:
    return web.json_response(lb.get_stats())


async def handle_slots(request: web.Request, lb: LoadBalancer) -> web.Response:
    backend = lb.get_least_loaded()
    proxy = ProxyBackend(backend)
    return await proxy.forward(request, "/slots")


async def handle_any(request: web.Request, lb: LoadBalancer) -> web.Response:
    return await handle_request(request, lb, request.path)


def create_app(backends: List[str]) -> web.Application:
    lb = LoadBalancer(backends)
    app = web.Application()
    app["lb"] = lb

    app.router.add_get("/health", lambda r: handle_health(r, lb))
    app.router.add_post("/v1/chat/completions", lambda r: handle_any(r, lb))
    app.router.add_post("/v1/completions", lambda r: handle_any(r, lb))
    app.router.add_post("/completion", lambda r: handle_any(r, lb))
    app.router.add_get("/slots", lambda r: handle_slots(r, lb))
    app.router.add_route("*", lambda r: handle_any(r, lb))

    return app


def main():
    parser = argparse.ArgumentParser(description="Load balancer for multi-GPU llama.cpp servers")
    parser.add_argument(
        "--backends",
        "-b",
        nargs="+",
        required=True,
        help="Backend server URLs (e.g., http://localhost:8080 http://localhost:8081)",
    )
    parser.add_argument("--port", "-p", type=int, default=8088, help="Load balancer port")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")

    args = parser.parse_args()

    logger.info(f"Starting load balancer with backends: {args.backends}")
    app = create_app(args.backends)
    web.run_app(app, host=args.host, port=args.port, access_log=logger)


if __name__ == "__main__":
    main()
