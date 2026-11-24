// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
/**
 * RNNoise noise suppression as an AudioProcessor (https://github.com/xiph/rnnoise).
 *
 * The engine is the optional peer dependency `@jitsi/rnnoise-wasm` (a synchronous WebAssembly build that runs in Node),
 * loaded on demand, so the core keeps zero runtime dependencies. Nothing is bundled.
 *
 * RNNoise works on 480-sample frames at 48 kHz, so run the pipeline with `targetSampleRate: 48000, frameMs: 10`.
 * `AudioPipeline` rejects a mismatch at construction. Use one processor per call: it keeps per-stream state.
 */
import { ProviderError } from "../errors.js";
import type { AudioProcessor } from "../audio-pipeline.js";

export const RNNOISE = "rnnoise";
export const RNNOISE_SAMPLE_RATE = 48000;
export const RNNOISE_FRAME_SAMPLES = 480;
const FRAME_BYTES = RNNOISE_FRAME_SAMPLES * 2;

/** The part of the Emscripten module this processor uses. Matches `@jitsi/rnnoise-wasm`'s sync build. */
export interface RnnoiseWasmModule {
  /** Re-read on every call: Emscripten replaces the view when memory grows. */
  readonly HEAPF32: Float32Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _rnnoise_create(model: number): number;
  _rnnoise_destroy(state: number): void;
  /** Returns the voice-activity probability, 0..1. Input and output are float32 holding 16-bit-scale samples. */
  _rnnoise_process_frame(state: number, out: number, input: number): number;
}

export type RnnoiseModuleFactory = () => RnnoiseWasmModule | Promise<RnnoiseWasmModule>;

export interface RnnoiseOptions {
  /** Supply your own module loader instead of importing `@jitsi/rnnoise-wasm`. */
  readonly moduleFactory?: RnnoiseModuleFactory;
}

interface Engine {
  readonly module: RnnoiseWasmModule;
  readonly state: number;
  readonly inPtr: number;
  readonly outPtr: number;
}

async function importJitsi(): Promise<RnnoiseModuleFactory> {
  // A variable specifier keeps this a runtime-only import: the peer is optional and ships no types.
  const specifier = "@jitsi/rnnoise-wasm/dist/rnnoise-sync.js";
  const loaded = (await import(specifier)) as { default?: unknown };
  const candidate = typeof loaded.default === "function" ? loaded.default : (loaded.default as { default?: unknown } | undefined)?.default;
  if (typeof candidate !== "function") throw new Error("@jitsi/rnnoise-wasm did not export a module factory.");
  return candidate as RnnoiseModuleFactory;
}

export class RnnoiseProcessor implements AudioProcessor {
  readonly name = RNNOISE;
  readonly kind = "noise_suppression" as const;
  readonly sampleRate = RNNOISE_SAMPLE_RATE;
  readonly frameSamples = RNNOISE_FRAME_SAMPLES;
  private engine: Engine | null;
  private detail: string | null;
  private lastVad: number | null = null;
  private disposed = false;

  private constructor(engine: Engine | null, detail: string | null) {
    this.engine = engine;
    this.detail = detail;
  }

  /**
   * Loads the engine. Never throws for a missing or broken engine: it returns a processor whose `health()` reports
   * `active: false` (with the reason), so a pipeline never claims suppression that is not running.
   */
  static async create(options: RnnoiseOptions = {}): Promise<RnnoiseProcessor> {
    let module: RnnoiseWasmModule | null = null;
    let state = 0;
    let inPtr = 0;
    let outPtr = 0;
    try {
      const factory = options.moduleFactory ?? (await importJitsi());
      module = await factory();
      state = module._rnnoise_create(0);
      inPtr = module._malloc(FRAME_BYTES * 2);
      outPtr = module._malloc(FRAME_BYTES * 2);
      if (state === 0 || inPtr === 0 || outPtr === 0) throw new Error("RNNoise could not allocate its state.");
      return new RnnoiseProcessor({ module, state, inPtr, outPtr }, null);
    } catch (error) {
      if (module !== null) {
        if (state !== 0) module._rnnoise_destroy(state);
        if (inPtr !== 0) module._free(inPtr);
        if (outPtr !== 0) module._free(outPtr);
      }
      const reason = error instanceof Error ? error.message : String(error);
      return new RnnoiseProcessor(null, `RNNoise engine not loaded: ${reason}`);
    }
  }

  /** Voice-activity probability of the last processed frame, or null before the first. */
  get voiceProbability(): number | null {
    return this.lastVad;
  }

  health(): { active: boolean; detail?: string } {
    if (this.disposed) return { active: false, detail: "RNNoise processor was disposed." };
    return this.engine === null ? { active: false, detail: this.detail ?? "RNNoise engine not loaded." } : { active: true };
  }

  process(pcm: Uint8Array): Uint8Array {
    if (this.disposed) throw new ProviderError({ code: "session_closed", provider: RNNOISE, message: "RNNoise processor was disposed." });
    if (this.engine === null) {
      throw new ProviderError({ code: "invalid_configuration", provider: RNNOISE, message: this.detail ?? "RNNoise engine not loaded." });
    }
    if (pcm.length !== FRAME_BYTES) {
      throw new ProviderError({
        code: "malformed_message",
        provider: RNNOISE,
        message: `RNNoise needs ${RNNOISE_FRAME_SAMPLES} PCM16 samples (${FRAME_BYTES} bytes) per frame, got ${pcm.length} bytes.`,
      });
    }
    const { module, state, inPtr, outPtr } = this.engine;
    const input = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    const inBase = inPtr >> 2;
    let heap = module.HEAPF32;
    for (let i = 0; i < RNNOISE_FRAME_SAMPLES; i++) heap[inBase + i] = input.getInt16(i * 2, true);
    this.lastVad = module._rnnoise_process_frame(state, outPtr, inPtr);
    heap = module.HEAPF32;
    const out = new Uint8Array(FRAME_BYTES);
    const view = new DataView(out.buffer);
    const outBase = outPtr >> 2;
    for (let i = 0; i < RNNOISE_FRAME_SAMPLES; i++) {
      view.setInt16(i * 2, Math.max(-32768, Math.min(32767, Math.floor(heap[outBase + i]! + 0.5))), true);
    }
    return out;
  }

  /** Frees the engine's memory. The pipeline has no close hook, so call this when the call ends. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.engine !== null) {
      const { module, state, inPtr, outPtr } = this.engine;
      module._rnnoise_destroy(state);
      module._free(inPtr);
      module._free(outPtr);
      this.engine = null;
    }
  }
}
