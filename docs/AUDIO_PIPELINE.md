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
| A-law | not implemented | Adapters may report `alaw`; the pipeline raises `unsupported_format` |
| Resampling | done | Streaming linear interpolation, integer-exact across TS and Python. Fine for upsampling (8 → 16 kHz). Downsampling has no anti-alias filter, so the pipeline adds `downsampling_without_antialiasing` to `status().warnings` |
| Framing | done | Fixed `frameMs` (default 20 ms); a short tail on `flush()` is emitted with `isCompleteFrame: false` and is not processed |
| Timestamps | done | Output frames keep the input's `timestampKind` and advance by `frameMs` |
| Processor chain | done | Any number of `AudioProcessor`s, run in order |
| Noise-suppression engine | **optional, not bundled** | RNNoise processor in both packages, engine supplied by you: see "RNNoise" below. Other engines: see "Engines" |
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
| RNNoise | BSD-3-Clause | 48 kHz, 480 samples (10 ms) | C; Python and WASM bindings exist | **Processor implemented** (see "RNNoise" below). Needs a 48 kHz pipeline or a dedicated resample around it |
| DeepFilterNet | MIT / Apache-2.0 | 48 kHz | Rust/ONNX; Python package | Stronger suppression, heavier CPU |
| WebRTC Audio Processing (NS/AGC) | BSD-3-Clause | 10 ms frames, 8–48 kHz | C++; Python bindings vary | Mature; also includes AGC |
| DTLN | MIT | 16 kHz | TF-Lite/ONNX | Fits a 16 kHz pipeline; quality varies with model |

RNNoise is wrapped as an `AudioProcessor` (below). The other engines are not. **Still open:** STT word error rate with and without an engine, on real call audio. Until that is measured, nothing here claims an accuracy improvement.

## RNNoise (optional)

`RnnoiseProcessor` wraps [RNNoise](https://github.com/xiph/rnnoise) as an `AudioProcessor`. No engine is bundled and the core keeps zero runtime dependencies. You supply the engine, and the processor loads it on demand.

| | TypeScript | Python |
|---|---|---|
| Import | `@vasikmansuri/calling-agent-toolkit/rnnoise` | `calling_agent_toolkit.processors.rnnoise` |
| Engine | optional peer dependency [`@jitsi/rnnoise-wasm`](https://www.npmjs.com/package/@jitsi/rnnoise-wasm) (a synchronous WebAssembly build; the `@shiguredo/rnnoise-wasm` build refuses to load in Node) | native `librnnoise`, loaded with `ctypes`: set `RNNOISE_LIBRARY` or pass `library_path`, or install it where the system loader finds `rnnoise` |
| Create | `await RnnoiseProcessor.create()` | `RnnoiseProcessor()` |
| Release | `dispose()` | `dispose()` |

```ts
import { AudioPipeline } from "@vasikmansuri/calling-agent-toolkit";
import { RnnoiseProcessor } from "@vasikmansuri/calling-agent-toolkit/rnnoise";

const rnnoise = await RnnoiseProcessor.create();            // one per call: it keeps per-stream state
const pipeline = new AudioPipeline({ targetSampleRate: 48000, frameMs: 10, processors: [rnnoise] });
// ... pipeline.push(frame) for the call, then rnnoise.dispose() when it ends
```

- **Shape:** RNNoise takes 480 samples at 48 kHz, so the pipeline must run at `targetSampleRate: 48000` with `frameMs: 10`. A 16 kHz pipeline refuses the processor at construction. The 48 kHz output has to go to an STT session that accepts 48 kHz (Deepgram does; the OpenAI adapter takes 24 kHz only, so the two do not combine without a resampling stage).
- **Honest status:** if the engine is missing or fails to load, `create()` / the constructor does **not** throw. `health()` returns `active: false` with the reason, `process()` throws, and the pipeline bypasses the processor, so `status().noiseSuppressionActive` is false and `processor_bypassed` is reported. Audio keeps flowing unprocessed. A disposed processor reports inactive too.
- **Samples:** PCM16 in, PCM16 out, same length. The engine works on float samples at 16-bit scale. Output is rounded and clamped to the 16-bit range.
- **Telephony input:** the pipeline's linear-interpolation resampler takes 8 kHz audio to 48 kHz. How RNNoise behaves on audio upsampled from 8 kHz was not evaluated.
- **Not bit-exact across languages:** the engines differ (WebAssembly vs native), so the shared fixture `contracts/fixtures/audio/rnnoise.json` checks behaviour (noise-only input is strongly attenuated), not exact samples.
- **Licences:** RNNoise is BSD-3-Clause. `@jitsi/rnnoise-wasm` ships an Apache-2.0 `LICENSE` file but declares no `license` field in its `package.json`. Verify both before shipping.

### Measured cost

`benchmarks/typescript/rnnoise-bench.mjs`: 20,000 frames after 2,000 warm-up frames, five runs, Node 22.23.5, a shared cloud VM (Intel Xeon @ 2.10 GHz, 2 logical CPUs). Input is noise with signal in every frame (see below). These are not guarantees:

| Per 10 ms frame (TypeScript, WebAssembly engine) | Mean | Median | p99 | Max |
|---|---|---|---|---|
| `RnnoiseProcessor.process` | 551–598 µs | 530–548 µs | 858–1083 µs | 4.0–9.4 ms |
| `AudioPipeline` 8 kHz μ-law → 48 kHz + RNNoise | 526–545 µs | 506–513 µs | 844–990 µs | 3.5–8.4 ms |

CPU time per frame matched wall time (550–595 µs for `process`). One live stream therefore needs roughly **5.5–6% of one core** on this machine (processing time ÷ the 10 ms frame). The worst single frames reached 9.4 ms, close to the frame interval, probably from VM scheduling or garbage collection. Plan for jitter rather than for the mean.

- Benchmark with signal in the frames. All-zero frames cost about 66 µs on a fresh processor in a quick check, versus about 550 µs for noise, so a silent test input understates a live call.
- This is processing cost only. RNNoise's algorithmic delay was **not** measured.
- **Python was not measured:** the native library could not be loaded in the session that wrote this, so there is no Python latency or CPU figure.
- **STT word error rate was not measured:** it needs real call audio and an STT account. No accuracy or quality improvement is claimed.

## Measured cost (no processors)

Per 20 ms frame, measured on 2026-10-23 on an Apple M4. These are not guarantees:

| | TypeScript (Node 26) | Python 3.10 |
|---|---|---|
| decode + resample 8→16 kHz + framing | 7.9 µs | 85 µs |

See [TESTING.md](TESTING.md#benchmarks).
