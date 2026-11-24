// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
import { ProviderError } from "./errors.js";
import { SUPPORTED_SAMPLE_RATES, type AudioFormat, type AudioFrame } from "./types.js";

export function bytesPerSample(format: AudioFormat): number {
  const width = format.encoding === "pcm_s16le" ? 2 : 1;
  return width * format.channels;
}

/** Validates a payload size against its declared format. Does not inspect sample values. */
export function validateAudioPayload(
  format: AudioFormat,
  byteLength: number,
  maxBytes: number,
  provider = "core",
): void {
  if (!(SUPPORTED_SAMPLE_RATES as readonly number[]).includes(format.sampleRate)) {
    throw new ProviderError({
      code: "unsupported_format",
      provider,
      message: `Sample rate ${format.sampleRate} Hz is not supported; expected one of ${SUPPORTED_SAMPLE_RATES.join(", ")}.`,
    });
  }
  if (byteLength > maxBytes) {
    throw new ProviderError({
      code: "buffer_overflow",
      provider,
      message: `Audio payload of ${byteLength} bytes exceeds the ${maxBytes}-byte limit.`,
    });
  }
  const width = bytesPerSample(format);
  if (byteLength === 0 || byteLength % width !== 0) {
    throw new ProviderError({
      code: "malformed_message",
      provider,
      message: `Audio payload of ${byteLength} bytes is not a whole number of ${width}-byte ${format.encoding} samples.`,
    });
  }
}

// ITU-T G.711 mu-law, bit-exact with CPython's audioop (contracts/fixtures/core/g711-mulaw.json).
// Encoding works on 14-bit samples (pcm16 >> 2, arithmetic shift), so small negatives round down.
const MULAW_BIAS = 0x84;
const MULAW_CLIP_14 = 8159;
const MULAW_SEGMENT_END_14 = [0x3f, 0x7f, 0xff, 0x1ff, 0x3ff, 0x7ff, 0xfff, 0x1fff] as const;

const MULAW_DECODE_TABLE: Int16Array = (() => {
  const table = new Int16Array(256);
  for (let i = 0; i < 256; i++) {
    const u = ~i & 0xff;
    const exponent = (u >> 4) & 0x07;
    const magnitude = ((((u & 0x0f) << 3) + MULAW_BIAS) << exponent) - MULAW_BIAS;
    table[i] = u & 0x80 ? -magnitude : magnitude;
  }
  return table;
})();

export function mulawToLinear(byte: number): number {
  return MULAW_DECODE_TABLE[byte & 0xff]!;
}

export function linearToMulaw(sample: number): number {
  let s = Math.max(-32768, Math.min(32767, Math.trunc(sample))) >> 2;
  let mask = 0xff;
  if (s < 0) {
    s = -s;
    mask = 0x7f;
  }
  if (s > MULAW_CLIP_14) s = MULAW_CLIP_14;
  s += MULAW_BIAS >> 2;
  const segment = MULAW_SEGMENT_END_14.findIndex((end) => s <= end);
  if (segment === -1) return 0x7f ^ mask;
  return ((segment << 4) | ((s >> (segment + 1)) & 0x0f)) ^ mask;
}

/** mu-law bytes -> little-endian PCM16 bytes. */
export function decodeMulaw(input: Uint8Array): Uint8Array {
  const out = new Uint8Array(input.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < input.length; i++) view.setInt16(i * 2, mulawToLinear(input[i]!), true);
  return out;
}

/** Little-endian PCM16 bytes -> mu-law bytes. */
export function encodeMulaw(pcm: Uint8Array): Uint8Array {
  if (pcm.length % 2 !== 0) {
    throw new ProviderError({ code: "malformed_message", provider: "core", message: "PCM16 input has an odd byte length." });
  }
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  const out = new Uint8Array(pcm.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = linearToMulaw(view.getInt16(i * 2, true));
  return out;
}

// ITU-T G.711 A-law, bit-exact with CPython's audioop (contracts/fixtures/core/g711-alaw.json).
// Encoding works on 13-bit samples (pcm16 >> 3, arithmetic shift), so small negatives round down.
// Even bits are inverted on the wire (XOR 0x55); the sign bit is set for non-negative samples.
const ALAW_SEGMENT_END_13 = [0x1f, 0x3f, 0x7f, 0xff, 0x1ff, 0x3ff, 0x7ff, 0xfff] as const;

const ALAW_DECODE_TABLE: Int16Array = (() => {
  const table = new Int16Array(256);
  for (let i = 0; i < 256; i++) {
    const a = i ^ 0x55;
    const segment = (a >> 4) & 0x07;
    const mantissa = ((a & 0x0f) << 4) + (segment === 0 ? 8 : 0x108);
    const magnitude = segment > 1 ? mantissa << (segment - 1) : mantissa;
    table[i] = a & 0x80 ? magnitude : -magnitude;
  }
  return table;
})();

export function alawToLinear(byte: number): number {
  return ALAW_DECODE_TABLE[byte & 0xff]!;
}

export function linearToAlaw(sample: number): number {
  let s = Math.max(-32768, Math.min(32767, Math.trunc(sample))) >> 3;
  let mask = 0xd5;
  if (s < 0) {
    s = -s - 1;
    mask = 0x55;
  }
  const segment = ALAW_SEGMENT_END_13.findIndex((end) => s <= end);
  if (segment === -1) return 0x7f ^ mask;
  return ((segment << 4) | ((s >> (segment < 2 ? 1 : segment)) & 0x0f)) ^ mask;
}

/** A-law bytes -> little-endian PCM16 bytes. */
export function decodeAlaw(input: Uint8Array): Uint8Array {
  const out = new Uint8Array(input.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < input.length; i++) view.setInt16(i * 2, alawToLinear(input[i]!), true);
  return out;
}

/** Little-endian PCM16 bytes -> A-law bytes. */
export function encodeAlaw(pcm: Uint8Array): Uint8Array {
  if (pcm.length % 2 !== 0) {
    throw new ProviderError({ code: "malformed_message", provider: "core", message: "PCM16 input has an odd byte length." });
  }
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  const out = new Uint8Array(pcm.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = linearToAlaw(view.getInt16(i * 2, true));
  return out;
}

/** Swaps each 16-bit sample between big- and little-endian. */
export function swapBytes16(input: Uint8Array): Uint8Array {
  const out = new Uint8Array(input.length);
  for (let i = 0; i + 1 < input.length; i += 2) {
    out[i] = input[i + 1]!;
    out[i + 1] = input[i]!;
  }
  return out;
}

/** Wire form of AudioFrame, matching core.schema.json. */
export type AudioFrameWire = Omit<AudioFrame, "payload" | "sequenceNumber" | "chunk"> & {
  readonly payloadBase64: string;
  readonly sequenceNumber?: number | null;
  readonly chunk?: number | null;
};

export function frameToWire(frame: AudioFrame): AudioFrameWire {
  const { payload, ...rest } = frame;
  return { ...rest, payloadBase64: Buffer.from(payload).toString("base64") };
}

export function frameFromWire(wire: AudioFrameWire): AudioFrame {
  const { payloadBase64, ...rest } = wire;
  return {
    ...rest,
    sequenceNumber: wire.sequenceNumber ?? null,
    chunk: wire.chunk ?? null,
    payload: new Uint8Array(Buffer.from(payloadBase64, "base64")),
  };
}
