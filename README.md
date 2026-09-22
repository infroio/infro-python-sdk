# infro

[![PyPI](https://img.shields.io/pypi/v/infro)](https://pypi.org/project/infro/)
[![CI](https://github.com/infroio/python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/infroio/python-sdk/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/infro)](https://pypi.org/project/infro/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-py.typed-blue.svg)](https://peps.python.org/pep-0561/)

The official Python client for the [INFRO](https://infro.io) API — one endpoint
for text, image, video and audio models.

```bash
pip install infro
```

```python
from infro import Infro

infro = Infro()  # reads INFRO_API_KEY

image = infro.images.generate(
    model="bfl/flux-2-pro",
    prompt="a lighthouse in fog, 35mm",
)
print(image["data"][0]["url"])
print(f"cost: ${image['usage']['cost']}")
```

## Why this and not the OpenAI SDK

For text, either works — INFRO serves an OpenAI-compatible `/v1/chat/completions`
and any client that lets you override the base URL will talk to it. What the
OpenAI SDK cannot express is the rest of the platform:

- **Images, video and audio**, with `usage.cost` on every response.
- **Video as an async job**, with a signed webhook when the render lands.
- **INFRO's request extensions** — `routing`, `fallbacks`, `logging`,
  `metadata` — as ordinary keyword arguments rather than an untyped
  `extra_body` dictionary.
- **The error taxonomy as exceptions**, so "should I retry this?" is
  `except RateLimitError` and not a substring match on a message.

## Configuration

```python
infro = Infro(
    api_key="sk_infro_...",          # or INFRO_API_KEY in the environment
    base_url="https://api.infro.io/v1",
    timeout=600.0,                    # seconds, per attempt
    max_retries=2,                    # attempts *after* the first
)
```

The key falls back to `INFRO_API_KEY` because a key in source is a key in
version control.

## Text

Shaped exactly like the OpenAI SDK, so moving a codebase across is a rename:

```python
completion = infro.chat.completions.create(
    model="anthropic/claude-sonnet-5",
    messages=[{"role": "user", "content": "Hello"}],
)
print(completion["choices"][0]["message"]["content"])
print(completion["model"])   # the model that *served* — matters with fallbacks
print(completion["route"])   # "primary" or "standby_a" — a position, never a vendor
```

Streaming yields chunks and ends at `[DONE]`:

```python
for chunk in infro.chat.completions.create(
    model="anthropic/claude-sonnet-5",
    messages=[{"role": "user", "content": "Write a haiku about fog."}],
    stream=True,
):
    delta = chunk["choices"][0]["delta"].get("content")
    if delta:
        print(delta, end="", flush=True)
```

A stream that ends *without* `[DONE]` raises rather than returning quietly: the
gateway closes that way when a request fails after the first byte, so silence
would hand you a truncated answer that looks complete.

## Routing, fallbacks and cost control

```python
completion = infro.chat.completions.create(
    model="openai/gpt-5.6-terra",
    messages=[{"role": "user", "content": "Hello"}],
    routing={"policy": "fastest", "regions": ["us", "eu"]},
    fallbacks=["deepseek/deepseek-v4-flash"],
    metadata={"user": "u_42", "feature": "summariser"},
    logging=False,   # this request's content is never stored
)
```

`metadata` is echoed on the request record and exported as OpenTelemetry span
attributes, so per-feature cost attribution needs no extra instrumentation.

## Video

Renders take minutes, so they are jobs. A webhook is the production path:

```python
job = infro.videos.create(
    model="kuaishou/kling-o3",
    prompt="slow push-in on a lighthouse in fog",
    duration_seconds=6,
    webhook={"url": "https://api.example.com/hooks/infro"},
)
print(job["id"], job["status"])
```

In a script or a test, where there is nowhere for a webhook to land, poll:

```python
finished = infro.jobs.wait_for(job["id"])
print(finished["output"]["url"])
```

`wait_for` raises on a failed render rather than returning it, because a caller
who awaited "the finished video" will use whatever comes back as one.

## Audio

```python
audio = infro.audio.speech(
    model="elevenlabs/eleven-v3",
    input="The lighthouse keeper watched the fog roll in.",
    voice="rachel",
)
open("out.mp3", "wb").write(audio)

transcript = infro.audio.transcribe(
    model="openai/whisper-large-v3",
    file="https://example.com/recording.mp3",
)
print(transcript["text"])
```

## Errors

```python
from infro import BudgetExceededError, RateLimitError, UpstreamError

try:
    infro.chat.completions.create(model="...", messages=[...])
except BudgetExceededError as error:
    ...          # top up; retrying will not help
except RateLimitError as error:
    ...          # already retried for you; error.retry_after is the server's own hint
except UpstreamError as error:
    print(error.request_id)   # the first thing support will ask for
```

Every exception carries `status`, `type`, `code`, `request_id` and `retryable`.
`retryable` is keyed on the HTTP status, not the type, because the status is
what the docs promise and what a proxy preserves.

## Retries

Only the four statuses the docs name as retryable — `408`, `429`, `502`, `503` —
and always with an `Idempotency-Key`, so a retry of a request the gateway
already accepted returns the first response instead of doing the work twice.
`Retry-After` wins over the computed backoff; jitter stops a fleet that failed
together from retrying together.

Streams are never retried: past the first byte the caller has already seen part
of an answer.

## Async

Every method has an `await`able twin with the same signature:

```python
from infro import AsyncInfro

async with AsyncInfro() as infro:
    completion = await infro.chat.completions.create(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "Hello"}],
    )
```

## Anything this client does not model

The gateway grows faster than a client library, so there is an escape hatch that
keeps the authentication and the retries:

```python
data = infro.request("GET", "/usage?from=2026-08-01")
```

## Documentation

<https://infro.io/docs> — the API reference, the error table, and the routing,
fallback and privacy semantics this client is a thin shell over.
