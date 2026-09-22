"""``infro`` — the official Python client for the INFRO API.

WHAT THIS EXISTS FOR, GIVEN THAT THE OPENAI SDK ALREADY WORKS
-------------------------------------------------------------

Text is genuinely covered by any OpenAI-compatible client, and the docs say so.
What no OpenAI client can express is the rest of the platform: an image
generation whose response carries ``usage.cost``, a video render that is an
async job with a signed webhook, a transcription, the catalog, the key's own
balance — and INFRO's request extensions (``routing``, ``fallbacks``,
``logging``, ``metadata``), which through the OpenAI SDK are an untyped
``extra_body`` dictionary.

So this client's job is to make the multimodal surface first-class while
keeping ``chat.completions.create`` shaped exactly like the OpenAI one, so that
moving a codebase across is a rename rather than a rewrite.

    from infro import Infro

    infro = Infro()   # reads INFRO_API_KEY

    image = infro.images.generate(
        model="bfl/flux-2-pro",
        prompt="a lighthouse in fog, 35mm",
    )
    print(image["data"][0]["url"], image["usage"]["cost"])

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------

It does not model providers, because the gateway never names one. It does not
expose a provider selection field, because the gateway refuses it with
``provider_selection_unsupported``. And it does not retry a request without an
idempotency key — see ``_client.py`` for why that is the difference between a
retry and a double charge.
"""

from __future__ import annotations

from typing import Any, Mapping

import httpx

from ._client import DEFAULT_BASE_URL, VERSION, AsyncTransport, Transport
from ._errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    InfroError,
    BudgetExceededError,
    InternalError,
    InvalidRequestError,
    NoAvailableProviderError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    TimeoutError,
    UpstreamError,
)
from ._resources import (
    AsyncAudio,
    AsyncChat,
    AsyncImages,
    AsyncJobs,
    AsyncKeys,
    AsyncModels,
    AsyncRequests,
    AsyncVideos,
    Audio,
    Chat,
    Images,
    Jobs,
    Keys,
    Models,
    Requests,
    Videos,
)

__all__ = [
    "Infro",
    "AsyncInfro",
    "DEFAULT_BASE_URL",
    "__version__",
    "InfroError",
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "NotFoundError",
    "RateLimitError",
    "BudgetExceededError",
    "UpstreamError",
    "NoAvailableProviderError",
    "TimeoutError",
    "InternalError",
    "ConnectionError",
]

__version__ = VERSION


class Infro:
    """The synchronous client.

    ``api_key`` falls back to ``INFRO_API_KEY`` in the environment, because a
    key in source is a key in version control and the one place a client can
    make that the harder path is its constructor.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = Transport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            client=http_client,
        )

        self.chat = Chat(self._transport)
        self.images = Images(self._transport)
        self.videos = Videos(self._transport)
        self.jobs = Jobs(self._transport)
        self.audio = Audio(self._transport)
        self.models = Models(self._transport)
        self.keys = Keys(self._transport)
        self.requests = Requests(self._transport)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def request(self, method: str, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        """Reach an endpoint this client does not model.

        Present on purpose: the gateway will grow faster than a client library,
        and the alternative to this method is somebody re-implementing
        authentication and retries in application code.
        """
        return self._transport.request(method, path, json_body, **kwargs)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "Infro":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncInfro:
    """The asynchronous client. Same surface, ``await``ed."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncTransport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            client=http_client,
        )

        self.chat = AsyncChat(self._transport)
        self.images = AsyncImages(self._transport)
        self.videos = AsyncVideos(self._transport)
        self.jobs = AsyncJobs(self._transport)
        self.audio = AsyncAudio(self._transport)
        self.models = AsyncModels(self._transport)
        self.keys = AsyncKeys(self._transport)
        self.requests = AsyncRequests(self._transport)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    async def request(self, method: str, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        return await self._transport.request(method, path, json_body, **kwargs)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> "AsyncInfro":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
