# Release checklist

**Nothing is published.** Both packages are deliberately locked:
- npm: `"private": true` in `packages/typescript/package.json`;
- PyPI: the `Private :: Do Not Upload` classifier in `packages/python/pyproject.toml` (PyPI rejects uploads with it).

Remove those locks only after the owner explicitly approves a release.

## Before the first release

- [ ] Owner approval to publish, recorded in the release PR.
- [ ] Package names re-checked: `@vasikmansuri/calling-agent-toolkit` (npm) and `calling-agent-toolkit` (PyPI) were free on 2026-10-23.
- [ ] Version bumped consistently (start at `0.1.0`) in both packages and `CHANGELOG.md`.
- [ ] Live provider tests run with test accounts (see TESTING.md, "Planned"):
  - Twilio: the handshake signature URL.
  - Plivo: the L16 byte order, the WebSocket `stop` event, `playAudio` `sampleRate` typing and the `stream_id` response field.
  - Telnyx: timestamp units and L16 byte order.
  - Vonage: outbound frame size.
  - Deepgram: `KeepAlive` and `CloseStream`.
  - OpenAI: whether deltas are strictly incremental, the `input_audio_buffer.commit` and `error` behaviour, and the A-law/μ-law (`audio/pcma`, `audio/pcmu`) input rates, which the docs do not state.
- [ ] PROVIDER_COMPATIBILITY.md statuses updated from those results, with no row upgraded without evidence.
- [ ] Licence review: runtime dependencies are none. Dev dependencies (vitest, ajv, typescript, pytest, jsonschema, mypy, langchain-core) are not shipped.
- [ ] `npm pack` and `uv build` artifacts inspected: only `dist/` and `LICENSE`/`README.md` in npm; only the package and `py.typed` in the wheel.
- [ ] CI green on Node 20/22 and Python 3.10/3.13.

## Publishing (after approval)

1. Remove `"private": true` and the `Private :: Do Not Upload` classifier, and bump the versions.
2. Configure npm provenance and PyPI trusted publishing for this repository.
3. Run the **release** workflow manually with `dry_run: false`. It requires approval in the protected `release` environment.
4. Tag `vX.Y.Z` and publish GitHub release notes from CHANGELOG.md.
