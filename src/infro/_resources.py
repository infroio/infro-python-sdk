"""The endpoint surface.

Every method takes and returns plain dictionaries with the field names the API
actually uses. That is deliberate: the docs, the curl examples and this SDK all
spell ``duration_seconds`` the same way, so a reader can copy a field out of a
published example and it works. A client that renamed fields to look more
Pythonic would make the documentation unusable and would have to be updated
every time the API grew one.

INFRO's own extensions — ``routing``, ``fallbacks``, ``logging``, ``metadata`` —
are ordinary keyword arguments here rather than an ``extra_body`` dictionary,
which is the single largest ergonomic difference from driving the gateway
through the OpenAI SDK.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Iterator, Mapping

from ._client import AsyncTransport, Transport, iter_sse
from ._errors import APIError

__all__ = [
    "Completions",
    "Chat",
    "Images",
    "Videos",
    "Jobs",
    "Audio",
    "Models",
    "Keys",
    "Requests",
    "AsyncCompletions",
    "AsyncChat",
    "AsyncImages",
    "AsyncVideos",
    "AsyncJobs",
    "AsyncAudio",
    "AsyncModels",
    "AsyncKeys",
    "AsyncRequests",
]

#: Terminal job statuses. Anything else means "ask again".
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def _body(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unset keyword arguments rather than sending explicit nulls.

    An explicit ``null`` is a *value* to most APIs and means "unset this",
    which is not what a caller who omitted an argument meant.
    """
    return {key: value for key, value in params.items() if value is not None}


class Completions:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        routing: dict[str, Any] | None = None,
        fallbacks: list[str] | None = None,
        logging: bool | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """A chat completion, or an iterator of chunks when ``stream=True``."""
        body = _body(
            {
                "model": model,
                "messages": messages,
                "routing": routing,
                "fallbacks": fallbacks,
                "logging": logging,
                "metadata": metadata,
                **kwargs,
            }
        )

        if not stream:
            return self._transport.request("POST", "/chat/completions", body)

        body["stream"] = True
        # `include_usage` on by default, because `usage.cost` is the one number
        # a caller nearly always wants and it is only emitted when asked for.
        body["stream_options"] = {"include_usage": True, **kwargs.get("stream_options", {})}

        response = self._transport.raw(
            "POST",
            "/chat/completions",
            body,
            headers={"Accept": "text/event-stream"},
            # A stream that fails after the first byte cannot be re-routed and
            # must not be replayed: the caller has already seen part of an
            # answer, and a replay would show them a sentence starting twice.
            max_retries=0,
            stream=True,
        )
        return _iter_chunks(response)


def _iter_chunks(response: Any) -> Iterator[dict[str, Any]]:
    """Parse an SSE body into chunks, honouring the documented ending rules."""
    request_id = response.headers.get("x-infro-request-id")
    saw_done = False
    try:
        for payload in iter_sse(response.iter_lines()):
            if payload == "[DONE]":
                saw_done = True
                return
            chunk = json.loads(payload)
            if "error" in chunk:
                # Nothing can be re-routed past the first byte, so this is the
                # ending. Raised rather than yielded, because a caller iterating
                # chunks would otherwise have to inspect every one for a field
                # that is almost never there.
                error = chunk["error"]
                raise APIError(
                    error.get("message", "The stream failed."),
                    type=error.get("type"),
                    code=error.get("code"),
                    status=response.status_code,
                    request_id=request_id,
                )
            yield chunk

        if not saw_done:
            # The gateway documents closing without `[DONE]` when a request
            # fails mid-stream, so its absence is meaningful. Returning the
            # partial answer silently is how truncated text reaches a user as
            # though it were complete.
            raise APIError(
                "The stream ended before [DONE]. The response is incomplete — "
                "see https://infro.io/docs/api/streaming.",
                type="upstream_error",
                status=response.status_code,
                request_id=request_id,
            )
    finally:
        response.close()


class Chat:
    def __init__(self, transport: Transport) -> None:
        self.completions = Completions(transport)


class Images:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        n: int | None = None,
        size: str | None = None,
        response_format: str | None = None,
        routing: dict[str, Any] | None = None,
        fallbacks: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            "/images/generations",
            _body(
                {
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "response_format": response_format,
                    "routing": routing,
                    "fallbacks": fallbacks,
                    **kwargs,
                }
            ),
        )


class Videos:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        image: str | None = None,
        webhook: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
        fallbacks: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a render. Returns immediately with a queued job."""
        return self._transport.request(
            "POST",
            "/videos",
            _body(
                {
                    "model": model,
                    "prompt": prompt,
                    "duration_seconds": duration_seconds,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "image": image,
                    "webhook": webhook,
                    "routing": routing,
                    "fallbacks": fallbacks,
                    **kwargs,
                }
            ),
        )

    def retrieve(self, job_id: str) -> dict[str, Any]:
        return self._transport.request("GET", f"/videos/{_path(job_id)}")


class Jobs:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def retrieve(self, job_id: str) -> dict[str, Any]:
        return self._transport.request("GET", f"/jobs/{_path(job_id)}")

    def list(self, **query: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/jobs{_query(query)}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._transport.request("POST", f"/jobs/{_path(job_id)}/cancel")

    def wait_for(
        self,
        job_id: str,
        *,
        timeout: float = 1800.0,
        poll_interval: float = 3.0,
    ) -> dict[str, Any]:
        """Poll until the job reaches a terminal status.

        A convenience, and deliberately an unglamorous one: a webhook is the
        documented way to learn a render finished, and polling is what you do in
        a script or a test where there is nowhere for a webhook to land.

        Raises on a failed job rather than returning it, because a caller who
        awaited "the finished video" and got a failure dictionary will use it as
        though it were a video.
        """
        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            job = self.retrieve(job_id)
            status = job.get("status")

            if status == "succeeded":
                return job
            if status in _TERMINAL:
                error = job.get("error") or {}
                raise APIError(
                    error.get("message", f"Job {job_id} ended as {status}."),
                    type=error.get("type"),
                    code=error.get("code"),
                    status=502,
                    request_id=job_id,
                )

            if time.monotonic() >= deadline:
                raise APIError(
                    f"Job {job_id} did not finish within the wait timeout. It may "
                    "still be running — retrieve it, or use a webhook.",
                    type="timeout_error",
                    status=408,
                    request_id=job_id,
                )

            time.sleep(interval)
            interval = min(interval * 2, 30.0)


class Audio:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def speech(self, *, model: str, input: str, voice: str, **kwargs: Any) -> bytes:
        """Text to speech. Returns the audio bytes."""
        response = self._transport.raw(
            "POST",
            "/audio/speech",
            _body({"model": model, "input": input, "voice": voice, **kwargs}),
            headers={"Accept": "audio/*"},
        )
        return response.content

    def transcribe(self, *, model: str, file: str, **kwargs: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", "/audio/transcriptions", _body({"model": model, "file": file, **kwargs})
        )


class Models:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def list(self) -> dict[str, Any]:
        return self._transport.request("GET", "/models")

    def retrieve(self, model_id: str) -> dict[str, Any]:
        return self._transport.request("GET", f"/models/{model_id}")


class Keys:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def retrieve(self) -> dict[str, Any]:
        """What this key is, what it may spend, and what is left."""
        return self._transport.request("GET", "/key")


class Requests:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def retrieve(self, request_id: str) -> dict[str, Any]:
        return self._transport.request("GET", f"/requests/{_path(request_id)}")

    def list(self, **query: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/requests{_query(query)}")


# ------------------------------------------------------------------ #
# Async mirrors
# ------------------------------------------------------------------ #


class AsyncCompletions:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        routing: dict[str, Any] | None = None,
        fallbacks: list[str] | None = None,
        logging: bool | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        body = _body(
            {
                "model": model,
                "messages": messages,
                "routing": routing,
                "fallbacks": fallbacks,
                "logging": logging,
                "metadata": metadata,
                **kwargs,
            }
        )

        if not stream:
            return await self._transport.request("POST", "/chat/completions", body)

        body["stream"] = True
        body["stream_options"] = {"include_usage": True, **kwargs.get("stream_options", {})}

        response = await self._transport.raw(
            "POST",
            "/chat/completions",
            body,
            headers={"Accept": "text/event-stream"},
            max_retries=0,
            stream=True,
        )
        return _aiter_chunks(response)


async def _aiter_chunks(response: Any) -> AsyncIterator[dict[str, Any]]:
    request_id = response.headers.get("x-infro-request-id")
    saw_done = False
    parts: list[str] = []
    try:
        async for line in response.aiter_lines():
            if line == "":
                if not parts:
                    continue
                payload = "\n".join(parts)
                parts = []
                if payload == "[DONE]":
                    saw_done = True
                    return
                chunk = json.loads(payload)
                if "error" in chunk:
                    error = chunk["error"]
                    raise APIError(
                        error.get("message", "The stream failed."),
                        type=error.get("type"),
                        code=error.get("code"),
                        status=response.status_code,
                        request_id=request_id,
                    )
                yield chunk
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                parts.append(line[5:].lstrip())

        if not saw_done:
            raise APIError(
                "The stream ended before [DONE]. The response is incomplete — "
                "see https://infro.io/docs/api/streaming.",
                type="upstream_error",
                status=response.status_code,
                request_id=request_id,
            )
    finally:
        await response.aclose()


class AsyncChat:
    def __init__(self, transport: AsyncTransport) -> None:
        self.completions = AsyncCompletions(transport)


class AsyncImages:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def generate(self, *, model: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return await self._transport.request(
            "POST", "/images/generations", _body({"model": model, "prompt": prompt, **kwargs})
        )


class AsyncVideos:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, *, model: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return await self._transport.request(
            "POST", "/videos", _body({"model": model, "prompt": prompt, **kwargs})
        )

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._transport.request("GET", f"/videos/{_path(job_id)}")


class AsyncJobs:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._transport.request("GET", f"/jobs/{_path(job_id)}")

    async def list(self, **query: Any) -> dict[str, Any]:
        return await self._transport.request("GET", f"/jobs{_query(query)}")

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return await self._transport.request("POST", f"/jobs/{_path(job_id)}/cancel")

    async def wait_for(
        self,
        job_id: str,
        *,
        timeout: float = 1800.0,
        poll_interval: float = 3.0,
    ) -> dict[str, Any]:
        import asyncio

        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            job = await self.retrieve(job_id)
            status = job.get("status")

            if status == "succeeded":
                return job
            if status in _TERMINAL:
                error = job.get("error") or {}
                raise APIError(
                    error.get("message", f"Job {job_id} ended as {status}."),
                    type=error.get("type"),
                    code=error.get("code"),
                    status=502,
                    request_id=job_id,
                )

            if time.monotonic() >= deadline:
                raise APIError(
                    f"Job {job_id} did not finish within the wait timeout. It may "
                    "still be running — retrieve it, or use a webhook.",
                    type="timeout_error",
                    status=408,
                    request_id=job_id,
                )

            await asyncio.sleep(interval)
            interval = min(interval * 2, 30.0)


class AsyncAudio:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def speech(self, *, model: str, input: str, voice: str, **kwargs: Any) -> bytes:
        response = await self._transport.raw(
            "POST",
            "/audio/speech",
            _body({"model": model, "input": input, "voice": voice, **kwargs}),
            headers={"Accept": "audio/*"},
        )
        return response.content

    async def transcribe(self, *, model: str, file: str, **kwargs: Any) -> dict[str, Any]:
        return await self._transport.request(
            "POST", "/audio/transcriptions", _body({"model": model, "file": file, **kwargs})
        )


class AsyncModels:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/models")

    async def retrieve(self, model_id: str) -> dict[str, Any]:
        return await self._transport.request("GET", f"/models/{model_id}")


class AsyncKeys:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def retrieve(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/key")


class AsyncRequests:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def retrieve(self, request_id: str) -> dict[str, Any]:
        return await self._transport.request("GET", f"/requests/{_path(request_id)}")

    async def list(self, **query: Any) -> dict[str, Any]:
        return await self._transport.request("GET", f"/requests{_query(query)}")


def _path(value: str) -> str:
    """Percent-encode an id into a path segment.

    A path built by concatenation is a path an id can escape, and an id is
    caller-supplied.
    """
    from urllib.parse import quote

    return quote(value, safe="")


def _query(params: Mapping[str, Any]) -> str:
    from urllib.parse import urlencode

    pairs = {key: value for key, value in params.items() if value is not None}
    return f"?{urlencode(pairs)}" if pairs else ""
