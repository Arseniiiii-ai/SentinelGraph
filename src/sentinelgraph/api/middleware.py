"""Streaming HTTP request-size enforcement for application defense in depth."""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Reject declared or streamed bodies larger than a configured byte bound."""

    def __init__(self, app: ASGIApp, *, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"detail":"request body exceeds the configured byte limit"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length", b"")
        if raw_length.isdigit() and int(raw_length) > self.maximum_bytes:
            await self._reject(send)
            return
        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        received = 0
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            chunks.append(chunk)
            received += len(chunk)
            if received > self.maximum_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered
            if disconnected or delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)
