// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
/**
 * OpenAI Realtime transcription (wss://api.openai.com/v1/realtime?intent=transcription).
 * Docs: https://developers.openai.com/api/docs/guides/realtime-transcription,
 *       https://developers.openai.com/api/docs/guides/realtime-websocket,
 *       https://developers.openai.com/api/docs/api-reference/realtime-client-events/session/update,
 *       https://developers.openai.com/api/docs/api-reference/realtime-server-events/error,
 *       https://developers.openai.com/api/docs/api-reference/realtime-server-events/conversation/item/input_audio_transcription/failed
 * The host opens the WebSocket (with `Authorization: Bearer <key>`) and passes it in as an SttConnection,
 * so API keys never pass through this module.
 */
import { ProviderError } from "../errors.js";
import { MessageReader, type JsonObject } from "../parse.js";
import { BoundedQueue } from "../queue.js";
import type { SttCapabilities, SttResult, StreamingSttSession } from "../stt.js";
import type { AudioFormat, AudioFrame } from "../types.js";
import type { SttConnection } from "./deepgram.js";

export const OPENAI = "openai";

/** Server-side transcription endpoint. The docs say to omit `model` from the URL and set it in `session.update`. */
export const OPENAI_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription";

/** The only input OpenAI documents for `audio/pcm`: "Only a 24kHz sample rate is supported." */
const INPUT_FORMAT: AudioFormat = { encoding: "pcm_s16le", sampleRate: 24000, channels: 1 };

function requireInputFormat(format: AudioFormat): void {
  if (format.encoding !== INPUT_FORMAT.encoding || format.sampleRate !== INPUT_FORMAT.sampleRate || format.channels !== INPUT_FORMAT.channels) {
    throw new ProviderError({
      code: "unsupported_format",
      provider: OPENAI,
      message: `OpenAI transcription takes ${INPUT_FORMAT.encoding}/${INPUT_FORMAT.sampleRate} mono, got ${format.encoding}/${format.sampleRate}/${format.channels}ch. Set the pipeline's targetSampleRate to 24000.`,
    });
  }
}

export interface OpenAiTranscriptionOptions {
  /** Transcription model, e.g. one of the values listed under `audio.input.transcription.model` in the session reference. */
  readonly model: string;
  /** ISO-639-1 code (`audio.input.transcription.language`). Not allowed together with `languages`. */
  readonly language?: string;
  /** ISO-639-1 codes (`audio.input.transcription.languages`); only some models accept it. */
  readonly languages?: readonly string[];
  readonly prompt?: string;
  /** Omitted: field not sent. `null`: VAD off, the host commits turns. Otherwise the VAD type, with OpenAI's default settings. */
  readonly turnDetection?: "server_vad" | "semantic_vad" | null;
}

/** Builds the `session.update` event that configures a transcription session. Pure; send it with `OpenAiStreamingSession.configure`. */
export function openaiSessionUpdate(options: OpenAiTranscriptionOptions): JsonObject {
  if (options.model.length === 0) {
    throw new ProviderError({ code: "invalid_configuration", provider: OPENAI, message: "A transcription model is required." });
  }
  if (options.language !== undefined && options.languages !== undefined) {
    throw new ProviderError({
      code: "invalid_configuration",
      provider: OPENAI,
      message: "Pass either language or languages, not both: OpenAI says not to send the singular field with languages.",
    });
  }
  if (options.languages !== undefined && options.languages.length === 0) {
    throw new ProviderError({ code: "invalid_configuration", provider: OPENAI, message: "languages needs at least one code." });
  }
  const transcription: JsonObject = { model: options.model };
  if (options.language !== undefined) transcription["language"] = options.language;
  if (options.languages !== undefined) transcription["languages"] = [...options.languages];
  if (options.prompt !== undefined) transcription["prompt"] = options.prompt;
  const input: JsonObject = { format: { type: "audio/pcm", rate: INPUT_FORMAT.sampleRate }, transcription };
  if (options.turnDetection === null) input["turn_detection"] = null;
  else if (options.turnDetection !== undefined) input["turn_detection"] = { type: options.turnDetection };
  return { type: "session.update", session: { type: "transcription", audio: { input } } };
}

export function openaiCapabilities(): SttCapabilities {
  return {
    provider: OPENAI,
    interimResults: true,
    // The docs list no word timestamps or confidence for transcription events, so none are reported.
    wordTimestamps: false,
    confidence: false,
    diarization: false,
    inputFormats: [INPUT_FORMAT],
  };
}

/**
 * Translates OpenAI server events into SttResults.
 *
 * - `...input_audio_transcription.delta` -> `interim`. Deltas are appended per item for display only. OpenAI calls
 *   them "newly available transcript text" and does not say whether later deltas may revise earlier text, so the
 *   interim text is never committed.
 * - `...input_audio_transcription.completed` -> `segment_final` (key `item_id:content_index`) then `turn_end`.
 *   `transcript` is authoritative. Each completed item is one committed audio unit (a VAD turn, or what the host committed).
 * - `...input_audio_transcription.failed` and `error` throw a ProviderError. Other events yield [].
 * Confidence and start time are `null`: the events carry neither.
 */
export class OpenAiResultMapper {
  private readonly pending = new Map<string, string>();
  private readonly r = new MessageReader(OPENAI);

  map(message: string | JsonObject): SttResult[] {
    const msg = typeof message === "string" ? this.r.json(message, "type") : message;
    switch (msg["type"]) {
      case "conversation.item.input_audio_transcription.delta": {
        const key = this.itemKey(msg);
        const text = (this.pending.get(key) ?? "") + this.r.string(msg, "delta");
        this.pending.set(key, text);
        return [{ type: "interim", text, confidence: null }];
      }
      case "conversation.item.input_audio_transcription.completed": {
        const key = this.itemKey(msg);
        const text = this.r.string(msg, "transcript");
        this.pending.delete(key);
        return [{ type: "segment_final", text, confidence: null, segmentKey: key, startMs: null }, { type: "turn_end" }];
      }
      case "conversation.item.input_audio_transcription.failed": {
        const key = this.itemKey(msg);
        this.pending.delete(key);
        throw this.providerError(this.r.object(msg, "error"), `Transcription failed for item ${key}`);
      }
      case "error":
        throw this.providerError(this.r.object(msg, "error"), "OpenAI reported an error");
      default:
        return [];
    }
  }

  private itemKey(msg: JsonObject): string {
    return `${this.r.id(msg, "item_id")}:${this.r.int(msg, "content_index")}`;
  }

  private providerError(error: JsonObject, prefix: string): ProviderError {
    const detail = (key: string): string | undefined => (typeof error[key] === "string" ? error[key] : undefined);
    const providerCode = detail("code") ?? detail("type");
    return new ProviderError({
      code: "unknown",
      provider: OPENAI,
      message: `${prefix}: ${detail("message") ?? "no message"}`,
      ...(providerCode !== undefined ? { providerCode } : {}),
    });
  }
}

export interface OpenAiSessionOptions {
  readonly maxQueuedResults?: number;
  /**
   * Send `input_audio_buffer.commit` from `finish()`. Default true, which is what you need when turn detection is
   * off. With VAD on, set false if everything is already committed: committing an empty buffer makes OpenAI send an `error`.
   */
  readonly commitOnFinish?: boolean;
}

export class OpenAiStreamingSession implements StreamingSttSession {
  private readonly queue: BoundedQueue<SttResult>;
  private readonly mapper = new OpenAiResultMapper();
  private readonly commitOnFinish: boolean;

  constructor(
    private readonly connection: SttConnection,
    readonly format: AudioFormat,
    options: OpenAiSessionOptions = {},
  ) {
    requireInputFormat(format);
    this.commitOnFinish = options.commitOnFinish ?? true;
    this.queue = new BoundedQueue({ capacity: options.maxQueuedResults ?? 1000, overflow: "reject", provider: OPENAI });
  }

  /** Sends `session.update`. Call once after the socket opens, before any audio. */
  async configure(options: OpenAiTranscriptionOptions): Promise<void> {
    await this.connection.sendText(JSON.stringify(openaiSessionUpdate(options)));
  }

  /** Feed each text message from the OpenAI socket here. Throws a ProviderError for `error` and transcription `failed` events. */
  receive(message: string): void {
    for (const result of this.mapper.map(message)) this.queue.push(result);
  }

  async sendAudio(frame: AudioFrame): Promise<void> {
    const f = frame.format;
    if (f.encoding !== this.format.encoding || f.sampleRate !== this.format.sampleRate || f.channels !== this.format.channels) {
      throw new ProviderError({
        code: "unsupported_format",
        provider: OPENAI,
        message: `Session expects ${this.format.encoding}/${this.format.sampleRate}, got ${f.encoding}/${f.sampleRate}.`,
      });
    }
    await this.connection.sendText(JSON.stringify({ type: "input_audio_buffer.append", audio: Buffer.from(frame.payload).toString("base64") }));
  }

  /** Commits the audio buffered so far, ending the current turn when turn detection is off. */
  async commit(): Promise<void> {
    await this.connection.sendText(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }

  async finish(): Promise<void> {
    if (this.commitOnFinish) await this.commit();
  }

  results(): AsyncIterable<SttResult> {
    return this.queue;
  }

  async close(): Promise<void> {
    this.queue.close();
    await this.connection.close();
  }
}
