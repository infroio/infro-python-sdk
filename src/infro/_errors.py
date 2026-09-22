"""The documented error taxonomy, as exceptions a caller can catch.

The gateway publishes a closed set of ``error.type`` values with an HTTP status
for each. A client that raised one exception for all of them would make the
most common decision — "should I retry this?" — a substring match on a message,
which is exactly what breaks when a message is reworded.

So there is a class per family, and ``retryable`` is derived from the *status*
rather than the type: the status is what the docs key retry guidance on, and it
is what a proxy in front of INFRO preserves even if it rewrites the body.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
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
    "error_from_response",
]

#: The four statuses the docs name as retryable, and no others.
#:
#: Deliberately not ``>= 500``. A ``500 internal_error`` is a defect in the
#: gateway's own code: retrying it fails identically and turns one bug into a
#: retry storm. ``502`` and ``503`` are upstream conditions a second attempt
#: genuinely re-rolls.
RETRYABLE_STATUSES = frozenset({408, 429, 502, 503})

#: The one code that overrides its own status.
#:
#: ``no_provider_connected`` is a 503 saying the organization has no connected
#: provider able to serve the model. Unlike every other 503, a second attempt
#: cannot change that — it stays true until somebody connects a provider in the
#: console, which the message tells them to do. Retrying spends the whole
#: backoff to arrive at advice the first response already carried, and it is
#: the error a new organization is most likely to meet.
NEVER_RETRY_CODES = frozenset({"no_provider_connected"})


class InfroError(Exception):
    """Base class for everything this package raises."""


class APIError(InfroError):
    """A response the gateway actually produced."""

    def __init__(
        self,
        message: str,
        *,
        type: str | None = None,
        code: str | None = None,
        status: int,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.type = type or "unknown"
        self.code = code
        self.status = status
        #: ``req_…``. The first thing support asks for.
        self.request_id = request_id
        #: Seconds the server asked us to wait, when it said.
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """Whether trying again could plausibly succeed.

        Keyed on status, except for the codes in :data:`NEVER_RETRY_CODES`,
        which describe a condition only the caller can clear.
        """
        if self.code is not None and self.code in NEVER_RETRY_CODES:
            return False
        return self.status in RETRYABLE_STATUSES

    def __str__(self) -> str:
        parts = [self.message, f"(status={self.status}", f"type={self.type}"]
        if self.code:
            parts.append(f"code={self.code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " ".join(parts) + ")"


class AuthenticationError(APIError):
    """401. The key is missing, malformed, or revoked."""


class PermissionDeniedError(APIError):
    """403. The key is valid and not allowed to do this."""


class InvalidRequestError(APIError):
    """400. Something about the request itself."""


class NotFoundError(APIError):
    """404. No such model, job, or request."""


class RateLimitError(APIError):
    """429. Honour ``retry_after``; the client already does."""


class BudgetExceededError(APIError):
    """402. A budget you set has no room for this request.

    Not a funding problem: INFRO holds no balance and takes no part in what
    your provider charges. Some ceiling in your own organization — an
    organization, project, member or key budget — would be passed by this
    request, so it was refused before it ran. The message names which one, and
    the remedy is to raise or remove it rather than to pay anybody.
    """


class UpstreamError(APIError):
    """502. Every route for the model failed."""


class NoAvailableProviderError(APIError):
    """503. No route was eligible — a policy or an outage, not a bad request."""


class TimeoutError(APIError):  # noqa: A001 - deliberately shadows the builtin here
    """408. The upstream did not answer in time."""


class InternalError(APIError):
    """500. A defect on our side. Not retryable — it fails identically."""


class ConnectionError(InfroError):  # noqa: A001 - deliberately shadows the builtin
    """A failure that never reached the gateway: DNS, TLS, a dropped socket.

    Separate from :class:`APIError` because the remedies differ and so does the
    blame: there is no request id, no type, and nothing INFRO can tell you about
    it. Retryable, because a connection that failed to open may open.
    """

    retryable = True


_BY_TYPE: dict[str, type[APIError]] = {
    "authentication_error": AuthenticationError,
    "permission_denied": PermissionDeniedError,
    "invalid_request_error": InvalidRequestError,
    "not_found_error": NotFoundError,
    "rate_limit_exceeded": RateLimitError,
    "budget_exceeded": BudgetExceededError,
    "upstream_error": UpstreamError,
    "no_available_provider": NoAvailableProviderError,
    "timeout_error": TimeoutError,
    "internal_error": InternalError,
}

_BY_STATUS: dict[int, type[APIError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    402: BudgetExceededError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: TimeoutError,
    429: RateLimitError,
    500: InternalError,
    502: UpstreamError,
    503: NoAvailableProviderError,
}


def error_from_response(
    status: int,
    body: Any,
    headers: Mapping[str, str],
) -> APIError:
    """Build the right exception from a response the gateway produced.

    Falls back to the status when the body is not the documented envelope — a
    proxy or a load balancer in front of INFRO can return HTML, and the error
    still has to be usable.
    """
    envelope = body.get("error") if isinstance(body, dict) else None
    error_type = envelope.get("type") if isinstance(envelope, dict) else None
    message = (
        envelope.get("message")
        if isinstance(envelope, dict) and envelope.get("message")
        else f"INFRO request failed with status {status}"
    )

    cls = _BY_TYPE.get(error_type or "") or _BY_STATUS.get(status, APIError)

    retry_after_raw = _header(headers, "retry-after")
    try:
        retry_after = float(retry_after_raw) if retry_after_raw is not None else None
    except ValueError:
        retry_after = None

    return cls(
        message,
        type=error_type,
        code=envelope.get("code") if isinstance(envelope, dict) else None,
        status=status,
        request_id=_header(headers, "x-infro-request-id"),
        retry_after=retry_after,
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup, because not every mapping is one."""
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None
