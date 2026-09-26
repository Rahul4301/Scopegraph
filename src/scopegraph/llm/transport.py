"""Pooled JSON transport shared by compatible model providers."""

import asyncio
from typing import Any

import httpx


class ModelTransport:
    def __init__(
        self, *, timeout: float = 60, retries: int = 3, client: httpx.AsyncClient | None = None
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self._client = client
        self.calls = 0
        self.retry_count = 0
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {}

    async def post(self, url: str, *, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        self.last_usage = {}
        for attempt in range(self.retries):
            try:
                self.calls += 1
                response = await self._client.post(
                    url, json=payload, headers={"Authorization": f"Bearer {api_key}"}
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("Model response must be a JSON object")
                self.last_usage = {
                    key: value
                    for key, value in body.get("usage", {}).items()
                    if isinstance(value, int)
                }
                for key, value in self.last_usage.items():
                    self.total_usage[key] = self.total_usage.get(key, 0) + value
                return body
            except httpx.HTTPStatusError as exc:
                retryable = (
                    exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                )
                if not retryable or attempt + 1 == self.retries:
                    # Provider error bodies explain rejections (bad parameter, context
                    # length, content policy) and do not echo credentials.
                    detail = exc.response.text[:500].strip()
                    raise RuntimeError(
                        f"Model endpoint returned HTTP {exc.response.status_code}"
                        + (f": {detail}" if detail else "")
                    ) from None
                self.retry_count += 1
                retry_after = exc.response.headers.get("retry-after", "")
                delay = min(float(retry_after), 30.0) if retry_after.isdigit() else 0.5 * 2**attempt
                await asyncio.sleep(delay)
            except httpx.TransportError:
                if attempt + 1 == self.retries:
                    raise RuntimeError("Model endpoint connection failed after retries") from None
                self.retry_count += 1
                await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError("Model request exhausted its retry budget")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def close_provider(provider: object) -> None:
    """Close composed providers without imposing network lifecycle on test fakes."""
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()
    elif hasattr(provider, "provider"):
        await close_provider(provider.provider)
