// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Vasik Mansuri
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";
import {
  AUDIO_ENCODINGS,
  CAPABILITIES,
  PROVIDER_ERROR_CODES,
  ProviderError,
  decodeAlaw,
  decodeMulaw,
  encodeAlaw,
  encodeMulaw,
  frameFromWire,
  frameToWire,
  requireCapability,
  validateAudioPayload,
  type AudioFormat,
  type ProviderCapabilities,
} from "../src/index.js";

const contracts = fileURLToPath(new URL("../../../contracts/", import.meta.url));
const load = (path: string): any => JSON.parse(readFileSync(contracts + path, "utf8"));

const schema = load("schemas/core.schema.json");
const ajv = new Ajv2020({ strict: false, allErrors: true });
ajv.addSchema(schema);
const validatorFor = (def: string) => ajv.getSchema(`${schema.$id}#/$defs/${def}`)!;

describe("schema cases", () => {
  const cases = load("fixtures/core/schema-cases.json");
  for (const [def, group] of Object.entries<any>(cases)) {
    if (def === "description") continue;
    group.valid.forEach((doc: unknown, i: number) =>
      it(`${def} valid[${i}]`, () => {
        const validate = validatorFor(def);
        expect(validate(doc), JSON.stringify(validate.errors)).toBe(true);
      }),
    );
    group.invalid.forEach((c: { why: string; doc: unknown }) =>
      it(`${def} rejects: ${c.why}`, () => expect(validatorFor(def)(c.doc)).toBe(false)),
    );
  }
});

describe("enums match the schema", () => {
  it.each([
    ["AudioEncoding", AUDIO_ENCODINGS],
    ["Capability", CAPABILITIES],
    ["ProviderErrorCode", PROVIDER_ERROR_CODES],
  ])("%s", (def, values) => expect([...values]).toEqual(schema.$defs[def].enum));
});

describe("capability checks", () => {
  const fixture = load("fixtures/core/capability-checks.json");
  const caps = fixture.capabilities as ProviderCapabilities;
  for (const c of fixture.cases) {
    it(c.capability, () => {
      if (c.ok) return expect(() => requireCapability(caps, c.capability)).not.toThrow();
      try {
        requireCapability(caps, c.capability);
        expect.unreachable();
      } catch (err) {
        expect(err).toBeInstanceOf(ProviderError);
        expect((err as ProviderError).code).toBe(c.code);
        if (c.messageContains) expect((err as ProviderError).message).toContain(c.messageContains);
        expect(validatorFor("ProviderError")((err as ProviderError).toJSON())).toBe(true);
      }
    });
  }
});

describe("frame validation", () => {
  const fixture = load("fixtures/core/frame-validation.json");
  for (const c of fixture.cases) {
    it(`${c.format.encoding}/${c.format.sampleRate}/${c.format.channels}ch ${c.byteLength}B`, () => {
      const run = () => validateAudioPayload(c.format as AudioFormat, c.byteLength, fixture.maxBytes);
      if (c.ok) return expect(run).not.toThrow();
      expect(run).toThrow(expect.objectContaining({ code: c.code }));
    });
  }
});

describe("G.711 mu-law vectors", () => {
  const fixture = load("fixtures/core/g711-mulaw.json");
  it("decodes all 256 codes", () => {
    const pcm = decodeMulaw(Uint8Array.from(fixture.decode.map((_: number, i: number) => i)));
    const view = new DataView(pcm.buffer);
    expect(fixture.decode.map((_: number, i: number) => view.getInt16(i * 2, true))).toEqual(fixture.decode);
  });
  it("encodes reference samples", () => {
    const samples: number[] = fixture.encode.map((e: [number, number]) => e[0]);
    const pcm = new Uint8Array(samples.length * 2);
    const view = new DataView(pcm.buffer);
    samples.forEach((s, i) => view.setInt16(i * 2, s, true));
    expect([...encodeMulaw(pcm)]).toEqual(fixture.encode.map((e: [number, number]) => e[1]));
  });
});

describe("G.711 A-law vectors", () => {
  const fixture = load("fixtures/core/g711-alaw.json");
  const allCodes = Uint8Array.from(fixture.decode.map((_: number, i: number) => i));
  it("decodes all 256 codes", () => {
    const pcm = decodeAlaw(allCodes);
    const view = new DataView(pcm.buffer);
    expect(fixture.decode.map((_: number, i: number) => view.getInt16(i * 2, true))).toEqual(fixture.decode);
  });
  it("encodes reference samples", () => {
    const samples: number[] = fixture.encode.map((e: [number, number]) => e[0]);
    const pcm = new Uint8Array(samples.length * 2);
    const view = new DataView(pcm.buffer);
    samples.forEach((s, i) => view.setInt16(i * 2, s, true));
    expect([...encodeAlaw(pcm)]).toEqual(fixture.encode.map((e: [number, number]) => e[1]));
  });
  it("round-trips every code", () => {
    expect([...encodeAlaw(decodeAlaw(allCodes))]).toEqual([...allCodes]);
  });
  it("rejects odd-length PCM16", () => {
    expect(() => encodeAlaw(new Uint8Array(3))).toThrow(expect.objectContaining({ code: "malformed_message" }));
  });
});

describe("AudioFrame wire round-trip", () => {
  const frames = load("fixtures/core/schema-cases.json").AudioFrame.valid;
  frames.forEach((wire: any, i: number) =>
    it(`valid[${i}]`, () => {
      const back = frameToWire(frameFromWire(wire));
      expect(validatorFor("AudioFrame")(back)).toBe(true);
      expect(back.payloadBase64).toBe(wire.payloadBase64);
    }),
  );
});
