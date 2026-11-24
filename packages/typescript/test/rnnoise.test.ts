// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AudioPipeline, type AudioFrame } from "../src/index.js";
import { RnnoiseProcessor, type RnnoiseWasmModule } from "../src/processors/rnnoise.js";

const contracts = fileURLToPath(new URL("../../../contracts/", import.meta.url));
const fixture = JSON.parse(readFileSync(contracts + "fixtures/audio/rnnoise.json", "utf8"));
const FRAME_BYTES = fixture.frameSamples * 2;

function* noiseFrames(seedStart: number, amplitude: number, frames: number): Generator<Uint8Array> {
  let seed = seedStart;
  for (let f = 0; f < frames; f++) {
    const out = new Uint8Array(FRAME_BYTES);
    const view = new DataView(out.buffer);
    for (let i = 0; i < fixture.frameSamples; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      view.setInt16(i * 2, Math.floor((seed / 2 ** 32 - 0.5) * amplitude + 0.5), true);
    }
    yield out;
  }
}

const energy = (pcm: Uint8Array): number => {
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let sum = 0;
  for (let i = 0; i < pcm.length; i += 2) sum += view.getInt16(i, true) ** 2;
  return sum;
};

/** A stand-in for the WebAssembly module that records allocations and lets a test choose the engine's output. */
function fakeModule(transform: (sample: number) => number, options: { failCreate?: boolean; growOnProcess?: boolean } = {}) {
  const calls = { malloc: 0, free: [] as number[], created: 0, destroyed: [] as number[] };
  let heap = new Float32Array(4096);
  let next = 64;
  const module: RnnoiseWasmModule = {
    get HEAPF32() {
      return heap;
    },
    _malloc: (bytes) => {
      calls.malloc++;
      const ptr = next;
      next += bytes;
      return ptr;
    },
    _free: (ptr) => void calls.free.push(ptr),
    _rnnoise_create: () => {
      calls.created++;
      return options.failCreate ? 0 : 7;
    },
    _rnnoise_destroy: (state) => void calls.destroyed.push(state),
    _rnnoise_process_frame: (_state, out, input) => {
      const source = heap;
      if (options.growOnProcess) heap = new Float32Array(heap); // Emscripten replaces the view when memory grows
      for (let i = 0; i < 480; i++) heap[(out >> 2) + i] = transform(source[(input >> 2) + i]!);
      return 0.25;
    },
  };
  return { module, calls };
}

describe("RnnoiseProcessor with the real WebAssembly engine", () => {
  it("loads, declares the shape RNNoise needs, and reports active", async () => {
    const rnnoise = await RnnoiseProcessor.create();
    try {
      expect(rnnoise.health()).toEqual({ active: true });
      expect([rnnoise.name, rnnoise.kind, rnnoise.sampleRate, rnnoise.frameSamples]).toEqual(["rnnoise", "noise_suppression", fixture.sampleRate, fixture.frameSamples]);
    } finally {
      rnnoise.dispose();
    }
  });

  it("suppresses noise-only input (shared fixture)", async () => {
    const { seed, amplitude, warmupFrames, frames, maxEnergyRatio } = fixture.noiseOnly;
    const rnnoise = await RnnoiseProcessor.create();
    let inEnergy = 0;
    let outEnergy = 0;
    let n = 0;
    for (const frame of noiseFrames(seed, amplitude, frames)) {
      const out = rnnoise.process(frame);
      expect(out.byteLength).toBe(FRAME_BYTES);
      if (n++ >= warmupFrames) {
        inEnergy += energy(frame);
        outEnergy += energy(out);
      }
    }
    expect(outEnergy / inEnergy).toBeLessThan(maxEnergyRatio);
    expect(rnnoise.voiceProbability).toBeGreaterThanOrEqual(0);
    expect(rnnoise.voiceProbability).toBeLessThanOrEqual(1);
    rnnoise.dispose();
  });

  it("keeps state per instance and is deterministic", async () => {
    const a = await RnnoiseProcessor.create();
    const b = await RnnoiseProcessor.create();
    const frames = [...noiseFrames(3, 6000, 5)];
    const fromA = frames.map((f) => a.process(f));
    // b has seen nothing, so it must match a's first frame: a's earlier frames cannot leak into other instances
    expect([...b.process(frames[0]!)]).toEqual([...fromA[0]!]);
    const a2 = await RnnoiseProcessor.create();
    expect(frames.map((f) => [...a2.process(f)])).toEqual(fromA.map((f) => [...f]));
    for (const p of [a, b, a2]) p.dispose();
  });

  it("rejects frames that are not 480 samples", async () => {
    const rnnoise = await RnnoiseProcessor.create();
    for (const bytes of [0, 2, FRAME_BYTES - 2, FRAME_BYTES + 2]) {
      expect(() => rnnoise.process(new Uint8Array(bytes))).toThrow(expect.objectContaining({ code: "malformed_message", provider: "rnnoise" }));
    }
    rnnoise.dispose();
  });

  it("refuses to process after dispose, and dispose is idempotent", async () => {
    const rnnoise = await RnnoiseProcessor.create();
    rnnoise.dispose();
    rnnoise.dispose();
    expect(rnnoise.health().active).toBe(false);
    expect(() => rnnoise.process(new Uint8Array(FRAME_BYTES))).toThrow(expect.objectContaining({ code: "session_closed" }));
  });
});

describe("RnnoiseProcessor sample handling (fake engine)", () => {
  it("converts PCM16 to float, back with rounding, and clamps to the 16-bit range", async () => {
    const { module } = fakeModule((s) => s * 2.5);
    const rnnoise = await RnnoiseProcessor.create({ moduleFactory: () => module });
    const pcm = new Uint8Array(FRAME_BYTES);
    const view = new DataView(pcm.buffer);
    [1, -1, 3, -3, 20000, -20000, 0].forEach((s, i) => view.setInt16(i * 2, s, true));
    const out = new DataView(rnnoise.process(pcm).buffer);
    // 2.5 -> 3, -2.5 -> -2 (round half up), 7.5 -> 8, -7.5 -> -7, then clamped
    expect([0, 1, 2, 3, 4, 5, 6].map((i) => out.getInt16(i * 2, true))).toEqual([3, -2, 8, -7, 32767, -32768, 0]);
    expect(rnnoise.voiceProbability).toBe(0.25);
  });

  it("re-reads HEAPF32 after processing, because the engine's memory can grow", async () => {
    const { module } = fakeModule((s) => s + 1, { growOnProcess: true });
    const rnnoise = await RnnoiseProcessor.create({ moduleFactory: () => module });
    const pcm = new Uint8Array(FRAME_BYTES);
    new DataView(pcm.buffer).setInt16(0, 41, true);
    for (let round = 0; round < 3; round++) expect(new DataView(rnnoise.process(pcm).buffer).getInt16(0, true)).toBe(42);
  });

  it("frees the engine's memory on dispose", async () => {
    const { module, calls } = fakeModule((s) => s);
    const rnnoise = await RnnoiseProcessor.create({ moduleFactory: () => module });
    rnnoise.dispose();
    expect(calls.destroyed).toEqual([7]);
    expect(calls.free).toHaveLength(2);
  });
});

describe("RnnoiseProcessor when the engine is unavailable", () => {
  it("reports inactive with the reason when the module cannot load", async () => {
    const rnnoise = await RnnoiseProcessor.create({
      moduleFactory: () => {
        throw new Error("wasm file missing");
      },
    });
    expect(rnnoise.health()).toEqual({ active: false, detail: "RNNoise engine not loaded: wasm file missing" });
    expect(() => rnnoise.process(new Uint8Array(FRAME_BYTES))).toThrow(expect.objectContaining({ code: "invalid_configuration" }));
  });

  it("reports inactive and releases memory when the engine cannot allocate", async () => {
    const { module, calls } = fakeModule((s) => s, { failCreate: true });
    const rnnoise = await RnnoiseProcessor.create({ moduleFactory: () => module });
    expect(rnnoise.health().active).toBe(false);
    expect(rnnoise.health().detail).toMatch(/could not allocate/);
    expect(calls.free).toHaveLength(2);
  });
});

describe("RnnoiseProcessor in an AudioPipeline", () => {
  const frame = (payload: Uint8Array): AudioFrame => ({
    callId: "c", sessionId: "s", provider: "twilio", direction: "inbound", format: { encoding: "mulaw", sampleRate: 8000, channels: 1 },
    payload, timestampMs: 0, timestampKind: "stream_relative_ms", sequenceNumber: 1, chunk: 1, isCompleteFrame: false,
  });
  const callAudio = () => [...noiseFrames(9, 20000, 100)].map((f) => f.slice(0, 80)); // 80 mu-law bytes = 10 ms at 8 kHz

  it("runs at 48 kHz / 10 ms and reports suppression active", async () => {
    const rnnoise = await RnnoiseProcessor.create();
    const pipeline = new AudioPipeline({ targetSampleRate: 48000, frameMs: 10, processors: [rnnoise] });
    const out = callAudio().flatMap((payload) => pipeline.push(frame(payload)));
    expect(out.length).toBeGreaterThan(90);
    expect(out.every((f) => f.payload.byteLength === FRAME_BYTES && f.format.sampleRate === 48000)).toBe(true);
    expect(pipeline.status().noiseSuppressionActive).toBe(true);
    expect(pipeline.status().warnings).toEqual([]);
    rnnoise.dispose();
  });

  it("is refused by a 16 kHz pipeline instead of receiving the wrong frames", async () => {
    const rnnoise = await RnnoiseProcessor.create();
    expect(() => new AudioPipeline({ targetSampleRate: 16000, processors: [rnnoise] })).toThrow(/480 samples at 48000/);
    rnnoise.dispose();
  });

  it("never claims suppression when the engine is missing, and audio keeps flowing", async () => {
    const rnnoise = await RnnoiseProcessor.create({ moduleFactory: () => Promise.reject(new Error("not installed")) });
    const pipeline = new AudioPipeline({ targetSampleRate: 48000, frameMs: 10, processors: [rnnoise] });
    const out = callAudio().flatMap((payload) => pipeline.push(frame(payload)));
    expect(out.length).toBeGreaterThan(90);
    const status = pipeline.status();
    expect(status.noiseSuppressionActive).toBe(false);
    expect(status.warnings).toContain("processor_bypassed");
    expect(status.processors[0]).toMatchObject({ name: "rnnoise", active: false, bypassed: true });
  });
});
