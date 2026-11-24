// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
/**
 * Shared audio path between a media adapter and STT:
 * decode -> validate -> resample -> fixed framing -> processors (noise suppression, speaker isolation).
 * The pipeline reports exactly which processors ran; it never claims suppression that did not happen.
 */
import { decodeAlaw, decodeMulaw } from "./audio.js";
import { ProviderError } from "./errors.js";
import { SUPPORTED_SAMPLE_RATES, type AudioFormat, type AudioFrame, type MediaDirection, type TimestampKind } from "./types.js";

const PROVIDER = "audio_pipeline";

function configError(message: string): ProviderError {
  return new ProviderError({ code: "invalid_configuration", provider: PROVIDER, message });
}

export function pcm16ToSamples(bytes: Uint8Array): Int16Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Int16Array(bytes.byteLength >> 1);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true);
  return out;
}

export function samplesToPcm16(samples: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i++) view.setInt16(i * 2, samples[i]!, true);
  return out;
}

/**
 * Streaming linear-interpolation resampler, integer-exact so TypeScript and Python agree bit for bit.
 * Linear interpolation is adequate for upsampling telephony audio (8 kHz -> 16 kHz). It does not
 * low-pass filter, so downsampling can alias; plug in a better resampler for that case.
 */
export class LinearResampler {
  private buffer: number[] = [];
  private base = 0;
  private outIndex = 0;

  constructor(
    readonly fromRate: number,
    readonly toRate: number,
  ) {
    if (!Number.isInteger(fromRate) || !Number.isInteger(toRate) || fromRate <= 0 || toRate <= 0) {
      throw configError(`Invalid resample rates ${fromRate} -> ${toRate}.`);
    }
  }

  push(samples: ArrayLike<number>): number[] {
    for (let i = 0; i < samples.length; i++) this.buffer.push(samples[i]!);
    return this.drain(false);
  }

  flush(): number[] {
    const out = this.drain(true);
    this.buffer = [];
    return out;
  }

  private drain(final: boolean): number[] {
    const { fromRate: from, toRate: to } = this;
    const out: number[] = [];
    if (from === to) {
      out.push(...this.buffer);
      this.base += this.buffer.length;
      this.outIndex += this.buffer.length;
      this.buffer = [];
      return out;
    }
    for (;;) {
      const num = this.outIndex * from;
      const local = Math.floor(num / to) - this.base;
      const r = num % to;
      if (local >= this.buffer.length || (!final && local + 1 >= this.buffer.length)) break;
      const a = this.buffer[local]!;
      const b = local + 1 < this.buffer.length ? this.buffer[local + 1]! : a;
      out.push(Math.floor((2 * (a * (to - r) + b * r) + to) / (2 * to)));
      this.outIndex++;
    }
    const keepFrom = Math.floor((this.outIndex * from) / to) - this.base;
    if (keepFrom > 0) {
      const drop = Math.min(keepFrom, this.buffer.length);
      this.buffer.splice(0, drop);
      this.base += drop;
    }
    return out;
  }
}

export type ProcessorKind = "noise_suppression" | "speaker_isolation" | "other";

/** A frame processor, e.g. an RNNoise or DeepFilterNet binding. Input and output are PCM16 LE mono. */
export interface AudioProcessor {
  readonly name: string;
  readonly kind: ProcessorKind;
  readonly sampleRate: number;
  readonly frameSamples: number;
  process(pcm: Uint8Array): Uint8Array;
  /** Report false when the engine is loaded but not actually processing (e.g. model missing). */
  health?(): { active: boolean; detail?: string };
}

export interface ProcessorStatus {
  readonly name: string;
  readonly kind: ProcessorKind;
  readonly active: boolean;
  readonly bypassed: boolean;
  readonly error: string | null;
}

export type PipelineWarning =
  | "stacked_noise_suppression"
  | "downsampling_without_antialiasing"
  | "processor_bypassed";

export interface PipelineStatus {
  readonly targetFormat: AudioFormat;
  readonly frameSamples: number;
  readonly processors: readonly ProcessorStatus[];
  readonly noiseSuppressionActive: boolean;
  readonly speakerIsolationActive: boolean;
  readonly providerNoiseSuppression: boolean;
  readonly warnings: readonly PipelineWarning[];
}

export interface AudioPipelineOptions {
  readonly targetSampleRate: number;
  readonly frameMs?: number;
  /** Only frames of this direction are processed; others are ignored. Default inbound (the caller). */
  readonly direction?: MediaDirection;
  readonly processors?: readonly AudioProcessor[];
  /** bypass (default): a failing processor is skipped and reported; throw: fail the frame. */
  readonly onProcessorError?: "bypass" | "throw";
  /** Set when the provider already suppresses noise (e.g. Plivo noise_cancellation). */
  readonly providerNoiseSuppression?: boolean;
}

interface ProcessorSlot {
  readonly processor: AudioProcessor;
  bypassed: boolean;
  error: string | null;
}

export class AudioPipeline {
  readonly targetFormat: AudioFormat;
  readonly frameSamples: number;
  private readonly frameMs: number;
  private readonly direction: MediaDirection;
  private readonly slots: ProcessorSlot[];
  private readonly onProcessorError: "bypass" | "throw";
  private readonly providerNoiseSuppression: boolean;
  private resampler: LinearResampler | null = null;
  private pending: number[] = [];
  private template: AudioFrame | null = null;
  private emitted = 0;
  private downsampling = false;

  constructor(options: AudioPipelineOptions) {
    const rate = options.targetSampleRate;
    if (!(SUPPORTED_SAMPLE_RATES as readonly number[]).includes(rate)) throw configError(`Unsupported target rate ${rate} Hz.`);
    this.frameMs = options.frameMs ?? 20;
    this.frameSamples = (rate * this.frameMs) / 1000;
    if (!Number.isInteger(this.frameSamples) || this.frameSamples <= 0) {
      throw configError(`frameMs ${this.frameMs} does not give a whole number of samples at ${rate} Hz.`);
    }
    for (const p of options.processors ?? []) {
      if (p.sampleRate !== rate || p.frameSamples !== this.frameSamples) {
        throw configError(
          `Processor ${p.name} needs ${p.frameSamples} samples at ${p.sampleRate} Hz; the pipeline produces ${this.frameSamples} at ${rate} Hz.`,
        );
      }
    }
    this.targetFormat = { encoding: "pcm_s16le", sampleRate: rate, channels: 1 };
    this.direction = options.direction ?? "inbound";
    this.slots = (options.processors ?? []).map((processor) => ({ processor, bypassed: false, error: null }));
    this.onProcessorError = options.onProcessorError ?? "bypass";
    this.providerNoiseSuppression = options.providerNoiseSuppression ?? false;
  }

  /** Accepts one provider frame; returns zero or more complete, processed frames at the target format. */
  push(frame: AudioFrame): AudioFrame[] {
    if (frame.direction !== this.direction) return [];
    if (frame.format.channels !== 1) {
      throw new ProviderError({ code: "unsupported_format", provider: PROVIDER, message: "Only mono audio is supported." });
    }
    let samples: Int16Array;
    if (frame.format.encoding === "mulaw") samples = pcm16ToSamples(decodeMulaw(frame.payload));
    else if (frame.format.encoding === "alaw") samples = pcm16ToSamples(decodeAlaw(frame.payload));
    else if (frame.format.encoding === "pcm_s16le") samples = pcm16ToSamples(frame.payload);
    else {
      throw new ProviderError({
        code: "unsupported_format",
        provider: PROVIDER,
        message: `No codec for ${frame.format.encoding} yet.`,
      });
    }
    if (this.resampler === null) {
      this.resampler = new LinearResampler(frame.format.sampleRate, this.targetFormat.sampleRate);
      this.downsampling = frame.format.sampleRate > this.targetFormat.sampleRate;
      this.template = frame;
    } else if (this.resampler.fromRate !== frame.format.sampleRate) {
      throw new ProviderError({
        code: "protocol_violation",
        provider: PROVIDER,
        message: `Sample rate changed mid-stream from ${this.resampler.fromRate} to ${frame.format.sampleRate} Hz.`,
      });
    }
    for (const s of this.resampler.push(samples)) this.pending.push(s);
    return this.drainFrames();
  }

  /** Ends the stream: emits remaining complete frames, then any short tail unprocessed (isCompleteFrame false). */
  flush(): AudioFrame[] {
    if (this.resampler === null) return [];
    for (const s of this.resampler.flush()) this.pending.push(s);
    const out = this.drainFrames();
    if (this.pending.length > 0) {
      out.push(this.makeFrame(samplesToPcm16(this.pending), false));
      this.pending = [];
    }
    return out;
  }

  status(): PipelineStatus {
    const processors = this.slots.map(({ processor, bypassed, error }) => ({
      name: processor.name,
      kind: processor.kind,
      bypassed,
      error,
      active: !bypassed && (processor.health?.().active ?? true),
    }));
    const activeNs = processors.filter((p) => p.kind === "noise_suppression" && p.active).length;
    const warnings: PipelineWarning[] = [];
    if (activeNs > 1 || (activeNs > 0 && this.providerNoiseSuppression)) warnings.push("stacked_noise_suppression");
    if (this.downsampling) warnings.push("downsampling_without_antialiasing");
    if (processors.some((p) => p.bypassed)) warnings.push("processor_bypassed");
    return {
      targetFormat: this.targetFormat,
      frameSamples: this.frameSamples,
      processors,
      noiseSuppressionActive: activeNs > 0,
      speakerIsolationActive: processors.some((p) => p.kind === "speaker_isolation" && p.active),
      providerNoiseSuppression: this.providerNoiseSuppression,
      warnings,
    };
  }

  private drainFrames(): AudioFrame[] {
    const out: AudioFrame[] = [];
    while (this.pending.length >= this.frameSamples) {
      const chunk = this.pending.splice(0, this.frameSamples);
      out.push(this.makeFrame(this.process(samplesToPcm16(chunk)), true));
    }
    return out;
  }

  private process(pcm: Uint8Array): Uint8Array {
    let current = pcm;
    for (const slot of this.slots) {
      if (slot.bypassed) continue;
      try {
        const result = slot.processor.process(current);
        if (result.byteLength !== current.byteLength) {
          throw new Error(`returned ${result.byteLength} bytes for a ${current.byteLength}-byte frame`);
        }
        current = result;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (this.onProcessorError === "throw") {
          throw new ProviderError({ code: "unknown", provider: PROVIDER, message: `Processor ${slot.processor.name} failed: ${message}` });
        }
        slot.bypassed = true;
        slot.error = message;
      }
    }
    return current;
  }

  private makeFrame(payload: Uint8Array, complete: boolean): AudioFrame {
    const t = this.template!;
    const kind: TimestampKind = t.timestampMs === null ? "none" : t.timestampKind;
    const index = this.emitted++;
    return {
      callId: t.callId,
      sessionId: t.sessionId,
      provider: t.provider,
      direction: t.direction,
      format: this.targetFormat,
      payload,
      timestampMs: kind === "none" ? null : t.timestampMs! + index * this.frameMs,
      timestampKind: kind,
      sequenceNumber: null,
      chunk: index,
      isCompleteFrame: complete,
    };
  }
}
