"""The transport, which is where every expensive mistake in a client lives.

The three that matter, and that this suite exists to prevent:

* **Retrying something that should not be retried.** A ``400`` retried three
  times is three times the latency for the same failure; a billed ``POST``
  retried without an idempotency key is somebody charged twice for one render.
* **Losing the request id.** It is the first thing support asks for and the last
  thing anybody records.
* **Mis-framing a stream.** An SSE frame split across two network reads is
  normal, not exceptional, and a parser that assumes otherwise works perfectly
  in development.

The transport is driven through a stub ``httpx`` client rather than a live
server, so what is asserted is the *request the SDK built* — the headers, the
body, and how many times it was sent.
"""

from __future__ import annotations

import json

import httpx
import pytest

from infro import (
    AsyncInfro,
    Infro,
    BudgetExceededError,
    InvalidRequestError,
    NoAvailableProviderError,
    RateLimitError,
    UpstreamError,
)
from infro._client import backoff_seconds
from infro._errors import APIError, ConnectionError

KEY = "unit-test-key-not-a-secret"


def responder(*responses):
    """An httpx transport that replays canned responses and records requests."""
    sent: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if not queue:
            raise AssertionError("no canned response left")
        nxt = queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    return httpx.MockTransport(handler), sent


def ok(body, status=200, headers=None):
    return httpx.Response(
        status,
        json=body,
        headers={"X-INFRO-Request-Id": "req_TEST01", **(headers or {})},
    )


def failure(status, type_, message, headers=None):
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": type_, "code": None}},
        headers={"X-INFRO-Request-Id": "req_TEST01", **(headers or {})},
    )


def make(*responses, max_retries=0):
    transport, sent = responder(*responses)
    client = Infro(
        api_key=KEY,
        base_url="https://api.test/v1",
        max_retries=max_retries,
        http_client=httpx.Client(transport=transport),
    )
    return client, sent


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #


def test_refuses_to_start_without_a_key(monkeypatch):
    # A key missing at construction is a configuration mistake. Discovering it
    # as a 401 on the first customer request is the same mistake, later.
    monkeypatch.delenv("INFRO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="INFRO_API_KEY"):
        Infro()


def test_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("INFRO_API_KEY", KEY)
    assert Infro().base_url == "https://api.infro.io/v1"


def test_strips_a_trailing_slash_from_the_base_url():
    client = Infro(api_key=KEY, base_url="https://api.test/v1/")
    assert client.base_url == "https://api.test/v1"


# ------------------------------------------------------------------ #
# Headers
# ------------------------------------------------------------------ #


def test_authenticates_and_identifies_itself():
    client, sent = make(ok({"ok": True}))
    client.keys.retrieve()

    assert sent[0].headers["authorization"] == f"Bearer {KEY}"
    assert sent[0].headers["user-agent"] == "infro-python/0.1.2"


def test_carries_an_idempotency_key_on_a_write():
    # On the first attempt, not only on the retry: the gateway keys on it when
    # it *first* sees the request.
    client, sent = make(ok({"id": "req_1"}))
    client.images.generate(model="m", prompt="p")
    assert sent[0].headers.get("idempotency-key")


def test_does_not_put_one_on_a_read():
    client, sent = make(ok({"data": []}))
    client.models.list()
    assert "idempotency-key" not in sent[0].headers


def test_honours_a_caller_supplied_idempotency_key():
    client, sent = make(ok({"id": "job_1"}))
    client.request("POST", "/videos", {"model": "m"}, idempotency_key="mine-42")
    assert sent[0].headers["idempotency-key"] == "mine-42"


# ------------------------------------------------------------------ #
# Errors
# ------------------------------------------------------------------ #


def test_raises_the_class_that_matches_the_type():
    # The point of a class per family: catching the one that matters must not
    # be a substring match on a message.
    cases = [
        (402, "budget_exceeded", BudgetExceededError),
        (429, "rate_limit_exceeded", RateLimitError),
        (400, "invalid_request_error", InvalidRequestError),
        (502, "upstream_error", UpstreamError),
        (503, "no_available_provider", NoAvailableProviderError),
    ]
    for status, type_, expected in cases:
        client, _ = make(failure(status, type_, "nope"))
        with pytest.raises(expected):
            client.models.list()


def test_carries_the_message_and_request_id():
    client, _ = make(failure(402, "budget_exceeded", "Your balance is exhausted."))
    with pytest.raises(BudgetExceededError) as caught:
        client.models.list()

    assert caught.value.message == "Your balance is exhausted."
    assert caught.value.request_id == "req_TEST01"
    assert caught.value.status == 402


def test_retryability_is_keyed_on_status_not_type():
    assert APIError("", status=429).retryable
    assert APIError("", status=503).retryable
    assert not APIError("", status=400).retryable
    # A defect in the gateway's own code fails identically on a retry.
    assert not APIError("", status=500).retryable


def test_no_provider_connected_is_not_retried_though_it_is_a_503():
    # The one code that overrides its own status. Nothing about a second
    # attempt connects a provider account, and it is the first error most new
    # organizations meet.
    assert not APIError("", status=503, code="no_provider_connected").retryable
    # A plain 503 with no code is still retryable.
    assert APIError("", status=503, code=None).retryable


def test_survives_a_body_that_is_not_the_envelope():
    # A load balancer in front of INFRO can return HTML.
    client, _ = make(httpx.Response(502, text="<html>bad gateway</html>"))
    with pytest.raises(UpstreamError) as caught:
        client.models.list()
    assert "502" in str(caught.value)


def test_a_connection_failure_is_its_own_class():
    client, _ = make(httpx.ConnectError("no route to host"))
    with pytest.raises(ConnectionError):
        client.models.list()


def test_the_string_form_names_the_request_id():
    error = APIError("Rate limited", type="rate_limit_exceeded", status=429, request_id="req_A")
    assert "req_A" in str(error)
    assert "status=429" in str(error)


# ------------------------------------------------------------------ #
# Retrying
# ------------------------------------------------------------------ #


def test_retries_a_503_and_returns_the_eventual_success():
    client, sent = make(
        failure(503, "no_available_provider", "no route"),
        ok({"data": []}),
        max_retries=1,
    )
    assert client.models.list() == {"data": []}
    assert len(sent) == 2


def test_does_not_retry_a_400():
    client, sent = make(failure(400, "invalid_request_error", "bad model"), max_retries=3)
    with pytest.raises(InvalidRequestError):
        client.models.list()
    assert len(sent) == 1


def test_does_not_retry_a_402():
    # More credit is not a matter of waiting.
    client, sent = make(failure(402, "budget_exceeded", "exhausted"), max_retries=3)
    with pytest.raises(BudgetExceededError):
        client.models.list()
    assert len(sent) == 1


def test_reuses_the_same_idempotency_key_across_a_retry():
    # The point of the key. A second attempt carrying a different one is a
    # second render, and a second charge.
    client, sent = make(
        failure(502, "upstream_error", "flaky"),
        ok({"id": "req_1"}),
        max_retries=1,
    )
    client.images.generate(model="m", prompt="p")

    assert sent[0].headers["idempotency-key"] == sent[1].headers["idempotency-key"]


def test_gives_up_after_the_configured_attempts():
    client, sent = make(
        failure(502, "upstream_error", "a"),
        failure(502, "upstream_error", "b"),
        failure(502, "upstream_error", "c"),
        max_retries=2,
    )
    with pytest.raises(UpstreamError, match="c"):
        client.models.list()
    assert len(sent) == 3


# ------------------------------------------------------------------ #
# Backoff
# ------------------------------------------------------------------ #


def test_backoff_obeys_retry_after():
    # The server knows when capacity frees up; a guess that undershoots
    # produces a second 429.
    error = APIError("", status=429, retry_after=12)
    assert backoff_seconds(1, error) == 12.0


def test_backoff_caps_an_absurd_retry_after():
    error = APIError("", status=429, retry_after=86_400)
    assert backoff_seconds(1, error) == 60.0


def test_backoff_is_jittered_and_bounded():
    class Low:
        def random(self):
            return 0.0

    class High:
        def random(self):
            return 1.0

    assert backoff_seconds(4, None, Low()) < backoff_seconds(4, None, High())
    assert backoff_seconds(30, None, High()) <= 8.0


# ------------------------------------------------------------------ #
# Resources
# ------------------------------------------------------------------ #


def test_sends_infro_extensions_as_top_level_fields():
    # Top level, not headers and not a nested envelope — which is what keeps a
    # codebase that uses them working against another OpenAI-compatible backend
    # that simply ignores them.
    client, sent = make(ok({"id": "c", "choices": [], "usage": {"cost": 0}}))
    client.chat.completions.create(
        model="m",
        messages=[],
        routing={"policy": "fastest", "regions": ["eu"]},
        fallbacks=["deepseek/deepseek-v4-flash"],
        logging=False,
        metadata={"user": "u_1"},
    )

    body = json.loads(sent[0].content)
    assert body["routing"] == {"policy": "fastest", "regions": ["eu"]}
    assert body["fallbacks"] == ["deepseek/deepseek-v4-flash"]
    assert body["logging"] is False
    assert body["metadata"] == {"user": "u_1"}


def test_omits_unset_arguments_rather_than_sending_null():
    # An explicit null is a *value* to most APIs and means "unset this", which
    # is not what a caller who omitted an argument meant.
    client, sent = make(ok({"id": "req_1"}))
    client.images.generate(model="m", prompt="p")

    body = json.loads(sent[0].content)
    assert set(body) == {"model", "prompt"}


def test_generates_an_image_and_reports_its_cost():
    client, sent = make(
        ok({"id": "req_1", "data": [{"url": "https://cdn.infro.io/x.png"}], "usage": {"cost": 0.04}})
    )
    image = client.images.generate(model="bfl/flux-2-pro", prompt="a lighthouse")

    assert str(sent[0].url) == "https://api.test/v1/images/generations"
    assert image["usage"]["cost"] == 0.04


def test_submits_a_video_as_a_job():
    client, sent = make(ok({"id": "job_1", "status": "queued"}))
    job = client.videos.create(
        model="kuaishou/kling-o3",
        prompt="a lighthouse",
        duration_seconds=6,
        webhook={"url": "https://example.com/hook"},
    )

    assert job["status"] == "queued"
    assert json.loads(sent[0].content)["duration_seconds"] == 6


def test_escapes_a_job_id_into_the_path():
    # A path built by concatenation is a path an id can escape, and an id is
    # caller-supplied.
    client, sent = make(ok({"id": "j"}))
    client.jobs.retrieve("job_../../admin")
    assert "/admin" not in str(sent[0].url)


# ------------------------------------------------------------------ #
# Waiting for a job
# ------------------------------------------------------------------ #


def test_wait_for_polls_until_it_succeeds():
    client, sent = make(
        ok({"id": "j", "status": "running"}),
        ok({"id": "j", "status": "succeeded", "output": {"url": "https://cdn/x.mp4"}}),
    )
    job = client.jobs.wait_for("j", poll_interval=0.001)

    assert job["status"] == "succeeded"
    assert len(sent) == 2


def test_wait_for_raises_on_a_failed_job():
    # A caller who awaited "the finished video" and got a failure dictionary
    # will use it as though it were a video.
    client, _ = make(
        ok(
            {
                "id": "j",
                "status": "failed",
                "error": {"message": "the render failed", "type": "upstream_error", "code": None},
            }
        )
    )
    with pytest.raises(APIError, match="the render failed"):
        client.jobs.wait_for("j", poll_interval=0.001)


def test_wait_for_gives_up_and_says_the_job_may_still_be_running():
    client, _ = make(ok({"id": "j", "status": "running"}))
    with pytest.raises(APIError, match="may still be running"):
        client.jobs.wait_for("j", poll_interval=0.001, timeout=0)


# ------------------------------------------------------------------ #
# Streaming
# ------------------------------------------------------------------ #


def chunk(content: str) -> bytes:
    payload = {
        "id": "c",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def sse(*parts: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        stream=httpx.ByteStream(b"".join(parts)),
        headers={"Content-Type": "text/event-stream", "X-INFRO-Request-Id": "req_STREAM"},
    )


def test_streams_chunks_and_stops_at_done():
    client, _ = make(sse(chunk("Hel"), chunk("lo"), b"data: [DONE]\n\n"))
    chunks = list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["Hel", "lo"]


def test_ignores_keep_alive_comments():
    # A proxy is entitled to inject them; a parser that raised would fail only
    # behind a load balancer.
    client, _ = make(sse(b": keep-alive\n\n", chunk("a"), b"data: [DONE]\n\n"))
    assert len(list(client.chat.completions.create(model="m", messages=[], stream=True))) == 1


def test_raises_a_mid_stream_error_frame():
    failure_frame = (
        json.dumps({"error": {"message": "upstream died", "type": "upstream_error", "code": None}})
    )
    client, _ = make(sse(chunk("a"), f"data: {failure_frame}\n\n".encode()))

    with pytest.raises(APIError, match="upstream died"):
        list(client.chat.completions.create(model="m", messages=[], stream=True))


def test_treats_a_stream_that_ends_without_done_as_truncated():
    # The gateway documents closing without [DONE] on a mid-stream failure, so
    # its absence is meaningful. Returning the partial answer silently is how
    # truncated text reaches a user as though it were complete.
    client, _ = make(sse(chunk("half an answer")))
    with pytest.raises(APIError, match="incomplete"):
        list(client.chat.completions.create(model="m", messages=[], stream=True))


def test_asks_for_usage_on_a_stream():
    client, sent = make(sse(b"data: [DONE]\n\n"))
    list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert json.loads(sent[0].content)["stream_options"]["include_usage"] is True


# ------------------------------------------------------------------ #
# The async client
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_async_client_mirrors_the_sync_one():
    transport, sent = responder(ok({"id": "req_1", "usage": {"cost": 0.04}}))
    async with AsyncInfro(
        api_key=KEY,
        base_url="https://api.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    ) as client:
        image = await client.images.generate(model="m", prompt="p")

    assert image["usage"]["cost"] == 0.04
    assert sent[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.asyncio
async def test_async_client_raises_the_same_classes():
    transport, _ = responder(failure(429, "rate_limit_exceeded", "slow down"))
    async with AsyncInfro(
        api_key=KEY,
        base_url="https://api.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    ) as client:
        with pytest.raises(RateLimitError):
            await client.models.list()


@pytest.mark.asyncio
async def test_async_streaming_yields_chunks():
    transport, _ = responder(sse(chunk("Hi"), b"data: [DONE]\n\n"))
    async with AsyncInfro(
        api_key=KEY,
        base_url="https://api.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    ) as client:
        stream = await client.chat.completions.create(model="m", messages=[], stream=True)
        contents = [c["choices"][0]["delta"]["content"] async for c in stream]

    assert contents == ["Hi"]
