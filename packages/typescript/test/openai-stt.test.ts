// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AudioPipeline, EmailCapture, ProviderError, SttAudioBridge, TranscriptReconciler, type AudioFrame } from "../src/index.js";
import {
  OPENAI_TRANSCRIPTION_URL,
  OpenAiResultMapper,
  OpenAiStreamingSession,
  openaiCapabilities,
  openaiSessionUpdate,
} from "../src/providers/openai.js";
import type { SttConnection } from "../src/providers/deepgram.js";

const contracts = fileURLToPath(new URL("../../../contracts/", import.meta.url));
const fixture: any = JSON.parse(readFileSync(contracts + "fixtures/providers/openai/messages.json", "utf8"));

describe("OpenAI message mapping", () => {
  for (const c of fixture.sequences) {
    it(c.name, () => {
      const mapper = new OpenAiResultMapper();
      expect(c.messages.map((m: unknown) => mapper.map(JSON.stringify(m)))).toEqual(c.expect);
    });
  }

  for (const c of fixture.errors) {
    it(`rejects: ${c.name}`, () => {
      const run = () => new OpenAiResultMapper().map(c.raw ?? JSON.stringify(c.message));
      expect(run).toThrow(expect.objectContaining({ code: c.expectError, provider: "openai" }));
      if (c.expectProviderCode) expect(run).toThrow(expect.objectContaining({ providerCode: c.expectProviderCode }));
      if (c.expectMessage) expect(run).toThrow(c.expectMessage);
    });
  }

  it("accepts an already-parsed object", () => {
    expect(new OpenAiResultMapper().map({ type: "conversation.item.input_audio_transcription.delta", item_id: "i", content_index: 0, delta: "hi" })).toEqual([
      { type: "interim", text: "hi", confidence: null },
    ]);
  });

  it("drops an item's partial text when its transcription fails", () => {
    const mapper = new OpenAiResultMapper();
    const key = { item_id: "item_1", content_index: 0 };
    mapper.map({ type: "conversation.item.input_audio_transcription.delta", ...key, delta: "stale" });
    expect(() => mapper.map({ type: "conversation.item.input_audio_transcription.failed", ...key, error: { message: "x" } })).toThrow(ProviderError);
    expect(mapper.map({ type: "conversation.item.input_audio_transcription.delta", ...key, delta: "fresh" })).toEqual([{ type: "interim", text: "fresh", confidence: null }]);
  });

  it("never invents timestamps or confidence", () => {
    const out = fixture.sequences.flatMap((c: any) => {
      const mapper = new OpenAiResultMapper();
      return c.messages.flatMap((m: unknown) => mapper.map(JSON.stringify(m)));
    });
    expect(out.length).toBeGreaterThan(0);
    expect(out.every((r: any) => r.type === "turn_end" || r.confidence === null)).toBe(true);
    expect(out.filter((r: any) => r.type === "segment_final").every((r: any) => r.startMs === null)).toBe(true);
  });
});

describe("OpenAI session configuration", () => {
  it("uses the documented transcription endpoint without a model", () => {
    expect(OPENAI_TRANSCRIPTION_URL).toBe("wss://api.openai.com/v1/realtime?intent=transcription");
  });

  for (const c of fixture.sessionUpdates) {
    it(c.name, () => expect(openaiSessionUpdate(c.options)).toEqual(c.expect));
  }
  for (const c of fixture.sessionUpdateErrors) {
    it(`rejects: ${c.name}`, () => expect(() => openaiSessionUpdate(c.options)).toThrow(expect.objectContaining({ code: c.expectError })));
  }

  it("declares what it can honestly provide", () => {
    expect(openaiCapabilities()).toEqual({
      provider: "openai",
      interimResults: true,
      wordTimestamps: false,
      confidence: false,
      diarization: false,
      inputFormats: [{ encoding: "pcm_s16le", sampleRate: 24000, channels: 1 }],
    });
  });
});

class FakeSocket implements SttConnection {
  binary: Uint8Array[] = [];
  text: string[] = [];
  closed = false;
  async sendBinary(data: Uint8Array) {
    this.binary.push(data);
  }
  async sendText(data: string) {
    this.text.push(data);
  }
  async close() {
    this.closed = true;
  }
}

const pcm24k = { encoding: "pcm_s16le", sampleRate: 24000, channels: 1 } as const;
const frame = (format: AudioFrame["format"], payload: Uint8Array): AudioFrame => ({
  callId: "c", sessionId: "s", provider: "twilio", direction: "inbound", format, payload,
  timestampMs: 0, timestampKind: "stream_relative_ms", sequenceNumber: 1, chunk: 1, isCompleteFrame: false,
});

describe("OpenAiStreamingSession", () => {
  it("rejects formats other than 24 kHz mono PCM16", () => {
    for (const format of [{ encoding: "pcm_s16le", sampleRate: 16000, channels: 1 }, { encoding: "mulaw", sampleRate: 8000, channels: 1 }, { encoding: "pcm_s16le", sampleRate: 24000, channels: 2 }] as const) {
      expect(() => new OpenAiStreamingSession(new FakeSocket(), format)).toThrow(expect.objectContaining({ code: "unsupported_format" }));
    }
  });

  it("sends session.update, then base64 audio appends", async () => {
    const socket = new FakeSocket();
    const session = new OpenAiStreamingSession(socket, pcm24k);
    await session.configure({ model: "gpt-4o-transcribe", turnDetection: null });
    await session.sendAudio(frame(pcm24k, Uint8Array.from([1, 2, 3, 4])));
    expect(socket.binary).toEqual([]);
    expect(socket.text.map((t) => JSON.parse(t))).toEqual([
      openaiSessionUpdate({ model: "gpt-4o-transcribe", turnDetection: null }),
      { type: "input_audio_buffer.append", audio: "AQIDBA==" },
    ]);
  });

  it("rejects frames in the wrong format without sending", async () => {
    const socket = new FakeSocket();
    const session = new OpenAiStreamingSession(socket, pcm24k);
    await expect(session.sendAudio(frame({ encoding: "mulaw", sampleRate: 8000, channels: 1 }, new Uint8Array(1)))).rejects.toMatchObject({ code: "unsupported_format" });
    expect(socket.text).toEqual([]);
  });

  it("commits on finish by default, and only when asked otherwise", async () => {
    const a = new FakeSocket();
    await new OpenAiStreamingSession(a, pcm24k).finish();
    expect(a.text).toEqual(['{"type":"input_audio_buffer.commit"}']);
    const b = new FakeSocket();
    const vad = new OpenAiStreamingSession(b, pcm24k, { commitOnFinish: false });
    await vad.finish();
    expect(b.text).toEqual([]);
    await vad.commit();
    expect(b.text).toEqual(['{"type":"input_audio_buffer.commit"}']);
  });

  it("surfaces provider errors from receive() and keeps working afterwards", async () => {
    const session = new OpenAiStreamingSession(new FakeSocket(), pcm24k);
    expect(() => session.receive(JSON.stringify({ type: "error", error: { type: "server_error", message: "boom" } }))).toThrow(/boom/);
    session.receive(JSON.stringify({ type: "conversation.item.input_audio_transcription.completed", item_id: "i", content_index: 0, transcript: "ok" }));
    await session.close();
    const out = [];
    for await (const r of session.results()) out.push(r);
    expect(out.map((r) => r.type)).toEqual(["segment_final", "turn_end"]);
  });

  it("closes the connection", async () => {
    const socket = new FakeSocket();
    await new OpenAiStreamingSession(socket, pcm24k).close();
    expect(socket.closed).toBe(true);
  });
});

describe("audio -> OpenAI -> transcript -> email", () => {
  it("sends exactly the pipeline's 24 kHz frames, base64-encoded", async () => {
    const socket = new FakeSocket();
    const session = new OpenAiStreamingSession(socket, pcm24k, { commitOnFinish: false });
    const bridge = new SttAudioBridge(new AudioPipeline({ targetSampleRate: 24000 }), session, pcm24k);
    const silence = frame({ encoding: "mulaw", sampleRate: 8000, channels: 1 }, new Uint8Array(160).fill(0xff));
    await bridge.push(silence);
    await bridge.push(silence);
    await bridge.end();
    const sent = socket.text.map((t) => JSON.parse(t));
    expect(sent.every((m) => m.type === "input_audio_buffer.append")).toBe(true);
    expect(sent.map((m) => Buffer.from(m.audio, "base64").byteLength)).toEqual([960, 960]);
  });

  it("refuses a pipeline that does not produce 24 kHz", () => {
    const session = new OpenAiStreamingSession(new FakeSocket(), pcm24k);
    expect(() => new SttAudioBridge(new AudioPipeline({ targetSampleRate: 16000 }), session, pcm24k)).toThrow(ProviderError);
  });

  it("captures an email from streamed deltas and the completed transcript", async () => {
    const session = new OpenAiStreamingSession(new FakeSocket(), pcm24k);
    const D = "conversation.item.input_audio_transcription.delta";
    const spoken = "my email is john dot smith at gmail dot com";
    for (const delta of ["my email is john dot ", "smith at gmail dot com"]) session.receive(JSON.stringify({ type: D, item_id: "item_1", content_index: 0, delta }));
    session.receive(JSON.stringify({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_1", content_index: 0, transcript: spoken }));
    await session.close();
    const reconciler = new TranscriptReconciler({ tenantId: "t1", sessionId: "call-1", now: () => 0 });
    const capture = new EmailCapture();
    const results = [];
    for await (const r of session.results()) for (const e of reconciler.accept(r)) results.push(...(await capture.handle(e)));
    expect(results).toEqual([{ type: "candidate", email: "john.smith@gmail.com", literal: false, turnId: "turn-1", replacesConfirmed: false }]);
  });
});
