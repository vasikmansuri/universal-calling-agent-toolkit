// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
// RNNoise cost per 10 ms frame. Needs the optional engine (`npm i @jitsi/rnnoise-wasm`, already a dev dependency),
// and a build first. Run from the repository root after `npm run build` in packages/typescript:
//   node benchmarks/typescript/rnnoise-bench.mjs
// Prints measured numbers only; results depend on the machine and are not a performance guarantee.
// This measures processing cost, not RNNoise's algorithmic delay, and not speech quality or STT accuracy.
import os from "node:os";
import { AudioPipeline, encodeMulaw } from "../../packages/typescript/dist/index.js";
import { RnnoiseProcessor } from "../../packages/typescript/dist/processors/rnnoise.js";

const FRAME_MS = 10;
const FRAMES = 20_000;
const WARMUP = 2_000;

const rnnoise = await RnnoiseProcessor.create();
if (!rnnoise.health().active) {
  console.error(rnnoise.health().detail);
  process.exit(1);
}

let seed = 1;
const noise = new Uint8Array(960);
const view = new DataView(noise.buffer);
for (let i = 0; i < 480; i++) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  view.setInt16(i * 2, Math.floor((seed / 2 ** 32 - 0.5) * 6000 + 0.5), true);
}

function measure(name, run) {
  for (let i = 0; i < WARMUP; i++) run(i);
  const samples = new Float64Array(FRAMES);
  const cpuBefore = process.cpuUsage();
  const wallStart = process.hrtime.bigint();
  for (let i = 0; i < FRAMES; i++) {
    const t0 = process.hrtime.bigint();
    run(i);
    samples[i] = Number(process.hrtime.bigint() - t0) / 1000;
  }
  const wallMicros = Number(process.hrtime.bigint() - wallStart) / 1000;
  const cpu = process.cpuUsage(cpuBefore);
  const sorted = Float64Array.from(samples).sort();
  const at = (p) => sorted[Math.min(FRAMES - 1, Math.floor(FRAMES * p))];
  const meanMicros = samples.reduce((a, b) => a + b, 0) / FRAMES;
  return {
    name,
    frames: FRAMES,
    meanMicros: +meanMicros.toFixed(2),
    p50Micros: +at(0.5).toFixed(2),
    p95Micros: +at(0.95).toFixed(2),
    p99Micros: +at(0.99).toFixed(2),
    maxMicros: +sorted[FRAMES - 1].toFixed(2),
    cpuMicrosPerFrame: +((cpu.user + cpu.system) / FRAMES).toFixed(2),
    // share of one core one live stream needs: processing time per frame / frame duration
    coreSharePercent: +((meanMicros / (FRAME_MS * 1000)) * 100).toFixed(3),
    wallMicrosPerFrame: +(wallMicros / FRAMES).toFixed(2),
  };
}

// Always benchmark frames that carry signal: all-zero frames can be much cheaper (about 66 us on a fresh processor
// here, versus about 550 us for noise), which would understate what a live call costs.
const results = [measure("RnnoiseProcessor.process, noise (480 samples @ 48 kHz)", () => rnnoise.process(noise))];

const pipeline = new AudioPipeline({ targetSampleRate: 48000, frameMs: FRAME_MS, processors: [await RnnoiseProcessor.create()] });
// 64 distinct, non-silent 10 ms mu-law frames (80 bytes at 8 kHz), cycled
const mulawFrames = Array.from({ length: 64 }, (_, k) => {
  const pcm = new Uint8Array(160);
  const v = new DataView(pcm.buffer);
  for (let i = 0; i < 80; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    v.setInt16(i * 2, Math.floor((seed / 2 ** 32 - 0.5) * 6000 + 0.5) + k, true);
  }
  return encodeMulaw(pcm);
});
const frame = (i) => ({
  callId: "c", sessionId: "s", provider: "bench", direction: "inbound", format: { encoding: "mulaw", sampleRate: 8000, channels: 1 },
  payload: mulawFrames[i % 64], timestampMs: i * FRAME_MS, timestampKind: "stream_relative_ms", sequenceNumber: i, chunk: i, isCompleteFrame: false,
});
results.push(measure("AudioPipeline 8 kHz mu-law (noise) -> 48 kHz + RNNoise, per 10 ms", (i) => pipeline.push(frame(i))));
if (!pipeline.status().noiseSuppressionActive) throw new Error("RNNoise was bypassed; the pipeline number would not include it.");

console.log(JSON.stringify({ runtime: `node ${process.version}`, platform: `${os.platform()} ${os.arch()}`, cpu: os.cpus()[0]?.model, logicalCpus: os.cpus().length, results }, null, 2));
rnnoise.dispose();
