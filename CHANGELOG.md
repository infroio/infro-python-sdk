# Changelog

All three INFRO SDKs share one version line: a customer reading a changelog
should not have to work out which of three independent version numbers applies
to them.

## 0.1.1

- Link the package metadata to its public GitHub repository and issue tracker.

## 0.1.0

First release.

- `Infro` and `AsyncInfro` over one `httpx` implementation, so streaming
  behaves identically in both rather than being two clients that drift.
- `chat.completions.create` (and `stream=True`), `images.generate`,
  `videos.create`, `jobs.retrieve` / `jobs.wait`, `audio.speech`,
  `audio.transcribe`, `models.list`, `keys.retrieve` and `requests.retrieve` —
  each response carrying `usage.cost`.
- The documented error taxonomy as exception classes, so "should I retry this?"
  is an `except` clause rather than a substring match on a message.
- Retries only on 408, 429, 502 and 503, always carrying the `Idempotency-Key`
  sent with the first attempt — except `no_provider_connected`, the one 503 a
  retry can never clear, which fails immediately so the advice in its message
  arrives without a backoff in front of it. A stream is never retried, and a stream that
  ends without `[DONE]` raises rather than returning the partial answer.
- `request()` as an escape hatch for endpoints this client does not yet model.
- Inline types, with a `py.typed` marker so a type checker actually uses them.
