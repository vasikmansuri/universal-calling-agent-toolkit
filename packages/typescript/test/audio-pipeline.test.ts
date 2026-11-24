// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  AudioPipeline,
  LinearResampler,
  pcm16ToSamples,
  samplesToPcm16,
  type AudioFrame,
  type AudioProcessor,
} from "../src/index.js";

const contracts = fileURLToPath(new URL("../../../contracts/", import.meta.url));
const load = (path: string): any => JSON.parse(readFileSync(contracts + path, "utf8"));

describe("LinearResampler vectors", () => {
  for (const c of load("fixtures/audio/resample.json").cases) {
    it(c.name, () => {
      const oneShot = new LinearResampler(c.from, c.to);
      expect([...oneShot.push(c.input), ...oneShot.flush()]).toEqual(c.output);
      const streamed = new LinearResampler(c.from, c.to);
      const out: number[] = [];
      let start = 0;
      for (const end of [...c.chunks, c.input.length]) {
        out.push(...streamed.push(c.input.slice(start, end)));
        start = end;
      }
      expect([...out, ...streamed.flush()]).toEqual(c.output);
    });
  }

  it("streams long input identically to one-shot", () => {
    const input = Array.from({ length: 1000 }, (_, i) => Math.round(Math.sin(i / 7) * 20000));
    const a = new LinearResampler(8000, 16000);
    const whole = [...a.push(input), ...a.flush()];
    const b = new LinearResampler(8000, 16000);
    const parts: number[] = [];
    for (let i = 0; i < input.length; i += 37) parts.push(...b.push(input.slice(i, i + 37)));
    expect([...parts, ...b.flush()]).toEqual(whole);
    expect(whole).toHaveLength(2000);
  });
});

const frame = (payload: Uint8Array, overrides: Partial<AudioFrame> = {}): AudioFrame => ({
  callId: "c",
  sessionId: "s",
  provider: "test",
  direction: "inbound",
  format: { encoding: "mulaw", sampleRate: 8000, channels: 1 },
  payload,
  timestampMs: 100,
  timestampKind: "stream_relative_ms",
  sequenceNumber: 1,
  chunk: 1,
  isCompleteFrame: false,
  ...overrides,
});

const gain = (factor: number, kind: AudioProcessor["kind"] = "noise_suppression"): AudioProcessor => ({
  name: `gain${factor}`,
  kind,
  sampleRate: 16000,
  frameSamples: 320,
  process: (pcm) => samplesToPcm16(Array.from(pcm16ToSamples(pcm), (s) => s * factor)),
});

describe("AudioPipeline", () => {
  it("turns 8 kHz mu-law chunks into 20 ms 16 kHz PCM frames", () => {
    const pipeline = new AudioPipeline({ targetSampleRate: 16000 });
    const silence = new Uint8Array(160).fill(0xff);
    const out = [...pipeline.push(frame(silence)), ...pipeline.push(frame(silence)), ...pipeline.flush()];
    expect(out.map((f) => f.payload.byteLength)).toEqual([640, 640]);
    expect(out.every((f) => f.isCompleteFrame && f.format.sampleRate === 16000 && f.format.encoding === "pcm_s16le")).toBe(true);
    expect(out.map((f) => f.timestampMs)).toEqual([100, 120]);
  });

  it("decodes 8 kHz A-law chunks into 20 ms 16 kHz PCM frames", () => {
    const pipeline = new AudioPipeline({ targetSampleRate: 16000 });
    const alaw = { format: { encoding: "alaw", sampleRate: 8000, channels: 1 } } as const;
    const quietPositive = new Uint8Array(160).fill(0xd5); // A-law 0xD5 decodes to +8
    const out = [...pipeline.push(frame(quietPositive, alaw)), ...pipeline.push(frame(quietPositive, alaw)), ...pipeline.flush()];
    expect(out.map((f) => f.payload.byteLength)).toEqual([640, 640]);
    expect(out.every((f) => f.isCompleteFrame && f.format.sampleRate === 16000 && f.format.encoding === "pcm_s16le")).toBe(true);
    expect(out.flatMap((f) => [...pcm16ToSamples(f.payload)]).every((s) => s === 8)).toBe(true);
  });

  it("runs processors in order and reports them active", () => {
    const pipeline = new AudioPipeline({ targetSampleRate: 16000, processors: [gain(2)] });
    const pcm = samplesToPcm16(new Array(320).fill(10));
    const [out] = pipeline.push(frame(pcm, { format: { encoding: "pcm_s16le", sampleRate: 16000, channels: 1 } }));
    expect(Array.from(pcm16ToSamples(out!.payload))).toEqual(new Array(320).fill(20));
    expect(pipeline.status()).toMatchObject({ noiseSuppressionActive: true, warnings: [] });
  });

  it("bypasses a failing processor and stops claiming suppression", () => {
    const broken: AudioProcessor = { ...gain(1), name: "broken", process: () => { throw new Error("model missing"); } };
    const pipeline = new AudioPipeline({ targetSampleRate: 16000, processors: [broken] });
    const pcm = samplesToPcm16(new Array(320).fill(7));
    const [out] = pipeline.push(frame(pcm, { format: { encoding: "pcm_s16le", sampleRate: 16000, channels: 1 } }));
    expect(out!.payload).toEqual(pcm);
    const status = pipeline.status();
    expect(status.noiseSuppressionActive).toBe(false);
    expect(status.processors[0]).toMatchObject({ bypassed: true, active: false, error: "model missing" });
    expect(status.warnings).toContain("processor_bypassed");
  });

  it("can fail loudly instead of bypassing", () => {
    const broken: AudioProcessor = { ...gain(1), process: () => new Uint8Array(2) };
    const pipeline = new AudioPipeline({ targetSampleRate: 16000, processors: [broken], onProcessorError: "throw" });
    const pcm = samplesToPcm16(new Array(320).fill(7));
    expect(() => pipeline.push(frame(pcm, { format: { encoding: "pcm_s16le", sampleRate: 16000, channels: 1 } }))).toThrow(/failed/);
  });

  it("trusts processor health over its presence", () => {
    const idle: AudioProcessor = { ...gain(1), health: () => ({ active: false, detail: "engine not loaded" }) };
    expect(new AudioPipeline({ targetSampleRate: 16000, processors: [idle] }).status().noiseSuppressionActive).toBe(false);
  });

  it("warns about stacked noise suppression", () => {
    expect(new AudioPipeline({ targetSampleRate: 16000, processors: [gain(1)], providerNoiseSuppression: true }).status().warnings).toEqual([
      "stacked_noise_suppression",
    ]);
    expect(new AudioPipeline({ targetSampleRate: 16000, processors: [gain(1), gain(1)] }).status().warnings).toEqual([
      "stacked_noise_suppression",
    ]);
  });

  it("reports speaker isolation separately", () => {
    const status = new AudioPipeline({ targetSampleRate: 16000, processors: [gain(1, "speaker_isolation")] }).status();
    expect(status).toMatchObject({ noiseSuppressionActive: false, speakerIsolationActive: true });
  });

  it("rejects processors whose frame shape does not match", () => {
    const rnnoiseShape: AudioProcessor = { ...gain(1), sampleRate: 48000, frameSamples: 480 };
    expect(() => new AudioPipeline({ targetSampleRate: 16000, processors: [rnnoiseShape] })).toThrow(/480 samples at 48000/);
  });

  it("ignores other directions, rejects unknown codecs and rate changes", () => {
    const pipeline = new AudioPipeline({ targetSampleRate: 16000 });
    expect(pipeline.push(frame(new Uint8Array(160), { direction: "outbound" }))).toEqual([]);
    const unknown = { encoding: "opus", sampleRate: 8000, channels: 1 } as unknown as AudioFrame["format"];
    expect(() => pipeline.push(frame(new Uint8Array(160), { format: unknown }))).toThrow(/No codec/);
    pipeline.push(frame(new Uint8Array(160)));
    expect(() => pipeline.push(frame(new Uint8Array(320), { format: { encoding: "pcm_s16le", sampleRate: 16000, channels: 1 } }))).toThrow(
      /changed mid-stream/,
    );
  });

  it("flags downsampling and emits a short tail as incomplete", () => {
    const pipeline = new AudioPipeline({ targetSampleRate: 8000 });
    const out = [...pipeline.push(frame(samplesToPcm16(new Array(100).fill(1)), { format: { encoding: "pcm_s16le", sampleRate: 16000, channels: 1 } })), ...pipeline.flush()];
    expect(out).toHaveLength(1);
    expect(out[0]!.isCompleteFrame).toBe(false);
    expect(pipeline.status().warnings).toContain("downsampling_without_antialiasing");
  });
});
