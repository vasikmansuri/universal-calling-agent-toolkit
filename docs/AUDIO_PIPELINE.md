# Audio pipeline

`AudioPipeline` (TS) and `AudioPipeline` (Python) sit between a media adapter and STT:

```
AudioFrame (any adapter) → decode → validate → resample → 20 ms framing → processors → STT
```

The STT adapter receives exactly the pipeline's output. `SttAudioBridge` connects the two and refuses a format mismatch instead of silently transcoding a second time. A test proves the bytes sent to STT are the processed frames.

## What is implemented

| Stage | Status | Notes |
|---|---|---|
| G.711 μ-law decode/encode | done | Bit-exact with CPython `audioop` across all 256 codes |
| PCM16 LE | done | Native format of the pipeline |
| G.711 A-law decode/encode | done | Bit-exact with CPython `audioop` across all 256 codes; the pipeline decodes `alaw` (e.g. Telnyx PCMA, A-law WAV) |
| Resampling | done | Streaming linear interpolation, integer-exact across TS and Python. Fine for upsampling (8 → 16 kHz). Downsampling has no anti-alias filter, so the pipeline adds `downsampling_without_antialiasing` to `status().warnings` |
| Framing | done | Fixed `frameMs` (default 20 ms); a short tail on `flush()` is emitted with `isCompleteFrame: false` and is not processed |
| Timestamps | done | Output frames keep the input's `timestampKind` and advance by `frameMs` |
| Processor chain | done | Any number of `AudioProcessor`s, run in order |
| Noise-suppression engine | **not bundled** | See "Engines" below |
| Speaker isolation engine | **not bundled** | Separate `kind: "speaker_isolation"`; reported separately |

## Processors and honest status

```ts
interface AudioProcessor {
  name: string;
  kind: "noise_suppression" | "speaker_isolation" | "other";
  sampleRate: number;     // must equal the pipeline's target rate
  frameSamples: number;   // must equal the pipeline's frame size
  process(pcm: Uint8Array): Uint8Array;  // same length out as in
  health?(): { active: boolean; detail?: string };
}
```

- **Shape check:** a processor whose sample rate or frame size doesn't match is rejected at construction. RNNoise needs 480 samples at 48 kHz; a 16 kHz pipeline refuses it rather than feeding it the wrong frames.
- **Failure handling:** if a processor throws or returns the wrong length, it is **bypassed**. Audio continues unprocessed and `status()` reports `active: false`, `bypassed: true` and the error. Use `onProcessorError: "throw"` to fail loudly instead.
- **Reporting:** `status().noiseSuppressionActive` is true only when a noise-suppression processor is present, not bypassed, and its `health()` reports active.
- **Stacking:** `stacked_noise_suppression` is reported when two suppressors are active, or when the provider's own suppression (`providerNoiseSuppression: true`, for example Plivo `noise_cancellation`) is on together with a toolkit suppressor. Stacked suppressors erode quiet speech and spelled letters, which is exactly what email capture needs.

## Engines (evaluation, not yet integrated)

No engine ships yet. Each has native, model or runtime requirements that must be verified on target hardware before it is declared supported, and none were benchmarked in this session.

| Engine | Licence (verify before use) | Native rate / frame | Runtime | Notes |
|---|---|---|---|---|
| RNNoise | BSD-3-Clause | 48 kHz, 480 samples (10 ms) | C; Python and WASM bindings exist | Light CPU; needs a 48 kHz pipeline or a dedicated resample around it |
| DeepFilterNet | MIT / Apache-2.0 | 48 kHz | Rust/ONNX; Python package | Stronger suppression, heavier CPU |
| WebRTC Audio Processing (NS/AGC) | BSD-3-Clause | 10 ms frames, 8–48 kHz | C++; Python bindings vary | Mature; also includes AGC |
| DTLN | MIT | 16 kHz | TF-Lite/ONNX | Fits a 16 kHz pipeline; quality varies with model |

Recommended next step: wrap one engine as an `AudioProcessor` in an optional package (`calling-agent-toolkit[audio]` or `@vasikmansuri/calling-agent-toolkit-rnnoise`), then measure latency, CPU and STT word error rate on real call audio before claiming improvement.

## Measured cost (no processors)

Per 20 ms frame, measured on 2026-10-23 on an Apple M4. These are not guarantees:

| | TypeScript (Node 26) | Python 3.10 |
|---|---|---|
| decode + resample 8→16 kHz + framing | 7.9 µs | 85 µs |

See [TESTING.md](TESTING.md#benchmarks).
