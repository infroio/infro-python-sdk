"""The transport every resource is built on.

WHY ``httpx`` AND NOT ``requests``
----------------------------------

``httpx`` is the only mainstream Python HTTP client that gives a synchronous
and an asynchronous API over one implementation, and streaming a chat
completion needs both to behave identically. Building the async client on
``aiohttp`` and the sync one on ``requests`` would mean two retry loops, two
timeout semantics and two sets of bugs.

RETRIES ARE NARROW AND IDEMPOTENT
---------------------------------

Only the four statuses the docs name as retryable — 408, 429, 502, 503 — and
only with an ``Idempotency-Key``, so a retry of a request the gateway already
accepted returns the first response rather than doing the work twice. Retrying
a billed POST without one is how somebody is charged twice for a render they
asked for once.

``Retry-After`` wins over the computed backoff when the server sends it, and
jitter is applied so a fleet that failed together does not retry together.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Iterator, Mapping

import httpx

from ._errors import RETRYABLE_STATUSES, APIError, ConnectionError, error_from_response

__all__ = ["DEFAULT_BASE_URL", "VERSION", "Transport", "AsyncTransport", "backoff_seconds"]

#: Published production base URL. Overridable for staging and for tests.
DEFAULT_BASE_URL = "https://api.infro.io/v1"

VERSION = "0.1.0"

_DEFAULT_TIMEOUT = 600.0
_DEFAULT_MAX_RETRIES = 2


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("INFRO_API_KEY")
    if not key:
        raise ValueError(
            "No INFRO API key. Pass api_key= or set INFRO_API_KEY in the "
            "environment. Create a key at https://dash.infro.io."
        )
    return key


def backoff_seconds(
    attempt: int,
    error: Exception | None,
    rand: "random.Random | None" = None,
) -> float:
    """How long to wait before ``attempt``.

    ``Retry-After`` first, because the server knows when capacity frees up and a
    computed guess that undershoots produces a second 429. Otherwise exponential
    with full jitter — a fleet that failed together must not retry together,
    which is what turns a blip into a thundering herd.
    """
    if isinstance(error, APIError) and error.retry_after is not None and error.retry_after >= 0:
        return min(error.retry_after, 60.0)
    ceiling = min(0.5 * (2 ** (attempt - 1)), 8.0)
    draw = (rand or random).random()
    return ceiling * (0.5 + draw * 0.5)


class _Base:
    """Everything the sync and async transports agree on."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._api_key = _resolve_key(api_key)
        self.base_url = (base_url or os.environ.get("INFRO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._default_headers = dict(default_headers or {})

    def _headers(
        self,
        method: str,
        extra: Mapping[str, str] | None,
        idempotency_key: str | None,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": f"infro-python/{VERSION}",
            **self._default_headers,
            **dict(extra or {}),
        }
        if method not in ("GET", "HEAD"):
            headers.setdefault("Content-Type", "application/json")
            # On every write, not only on the retry: the gateway keys on this
            # when it *first* sees the request, so adding it on the second
            # attempt would be a different request as far as the server is
            # concerned.
            headers["Idempotency-Key"] = idempotency_key or str(uuid.uuid4())
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"


class Transport(_Base):
    """Synchronous transport."""

    def __init__(self, *, client: httpx.Client | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None

    def request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        stream: bool = False,
    ) -> Any:
        response = self.raw(
            method,
            path,
            json_body,
            max_retries=max_retries,
            headers=headers,
            idempotency_key=idempotency_key,
            stream=stream,
        )
        if stream:
            return response
        if response.status_code == 204:
            return None
        return response.json()

    def raw(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        attempts = self.max_retries if max_retries is None else max_retries
        request_headers = self._headers(method, headers, idempotency_key)
        url = self._url(path)
        last: Exception | None = None

        for attempt in range(attempts + 1):
            if attempt > 0:
                time.sleep(backoff_seconds(attempt, last))

            try:
                if stream:
                    # `send(stream=True)` so the body is not read into memory —
                    # a streamed completion has to be iterable as it arrives,
                    # which is the whole point of asking for one.
                    request = self._client.build_request(
                        method, url, headers=request_headers, json=json_body
                    )
                    response = self._client.send(request, stream=True)
                else:
                    response = self._client.request(
                        method, url, headers=request_headers, json=json_body
                    )
            except httpx.HTTPError as exc:
                error = ConnectionError(f"Could not reach {self.base_url}: {exc}")
                if attempt == attempts:
                    raise error from exc
                last = error
                continue

            if response.is_success:
                return response

            if stream:
                # The body has not been read yet, and the error message is in it.
                response.read()
            error = error_from_response(
                response.status_code, _safe_json(response), response.headers
            )
            if stream:
                response.close()

            if not error.retryable or attempt == attempts:
                raise error
            last = error

        raise last or ConnectionError("Request failed with no outcome")  # pragma: no cover

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class AsyncTransport(_Base):
    """Asynchronous transport. Same decisions, same order, `await`ed."""

    def __init__(self, *, client: httpx.AsyncClient | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or httpx.AsyncClient(timeout=self.timeout)
        self._owns_client = client is None

    async def request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        stream: bool = False,
    ) -> Any:
        response = await self.raw(
            method,
            path,
            json_body,
            max_retries=max_retries,
            headers=headers,
            idempotency_key=idempotency_key,
            stream=stream,
        )
        if stream:
            return response
        if response.status_code == 204:
            return None
        return response.json()

    async def raw(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        import asyncio

        attempts = self.max_retries if max_retries is None else max_retries
        request_headers = self._headers(method, headers, idempotency_key)
        url = self._url(path)
        last: Exception | None = None

        for attempt in range(attempts + 1):
            if attempt > 0:
                await asyncio.sleep(backoff_seconds(attempt, last))

            try:
                if stream:
                    request = self._client.build_request(
                        method, url, headers=request_headers, json=json_body
                    )
                    response = await self._client.send(request, stream=True)
                else:
                    response = await self._client.request(
                        method, url, headers=request_headers, json=json_body
                    )
            except httpx.HTTPError as exc:
                error = ConnectionError(f"Could not reach {self.base_url}: {exc}")
                if attempt == attempts:
                    raise error from exc
                last = error
                continue

            if response.is_success:
                return response

            if stream:
                await response.aread()
            error = error_from_response(
                response.status_code, _safe_json(response), response.headers
            )
            if stream:
                await response.aclose()

            if not error.retryable or attempt == attempts:
                raise error
            last = error

        raise last or ConnectionError("Request failed with no outcome")  # pragma: no cover

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is a normal proxy outcome
        return None


def iter_sse(lines: Iterator[str]) -> Iterator[str]:
    """Yield each ``data:`` payload from a stream of already-split lines.

    Frames are separated by a blank line and a ``data:`` value may span several
    of them, so this accumulates rather than treating one line as one frame.
    ``:`` comment lines are skipped, because a proxy is entitled to inject
    keep-alives and a parser that raised on one would fail only behind a load
    balancer.
    """
    parts: list[str] = []
    for line in lines:
        if line == "":
            if parts:
                yield "\n".join(parts)
                parts = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            parts.append(line[5:].lstrip())
    if parts:
        yield "\n".join(parts)
