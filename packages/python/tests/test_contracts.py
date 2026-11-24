# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from calling_agent_toolkit import (
    AUDIO_ENCODINGS,
    CAPABILITIES,
    PROVIDER_ERROR_CODES,
    AudioFormat,
    ProviderCapabilities,
    ProviderError,
    decode_alaw,
    decode_mulaw,
    encode_alaw,
    encode_mulaw,
    frame_from_wire,
    frame_to_wire,
    require_capability,
    validate_audio_payload,
)

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"


def load(path: str) -> Any:
    return json.loads((CONTRACTS / path).read_text())


SCHEMA = load("schemas/core.schema.json")
REGISTRY = Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))


def validator(definition: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"{SCHEMA['$id']}#/$defs/{definition}"}, registry=REGISTRY)


SCHEMA_CASES = {k: v for k, v in load("fixtures/core/schema-cases.json").items() if k != "description"}


@pytest.mark.parametrize(
    ("definition", "doc"), [(d, doc) for d, g in SCHEMA_CASES.items() for doc in g["valid"]]
)
def test_schema_accepts(definition: str, doc: Any) -> None:
    validator(definition).validate(doc)


@pytest.mark.parametrize(
    ("definition", "case"),
    [(d, c) for d, g in SCHEMA_CASES.items() for c in g["invalid"]],
    ids=lambda v: v["why"] if isinstance(v, dict) else v,
)
def test_schema_rejects(definition: str, case: dict[str, Any]) -> None:
    assert not validator(definition).is_valid(case["doc"])


@pytest.mark.parametrize(
    ("definition", "values"),
    [("AudioEncoding", AUDIO_ENCODINGS), ("Capability", CAPABILITIES), ("ProviderErrorCode", PROVIDER_ERROR_CODES)],
)
def test_enums_match_schema(definition: str, values: tuple[str, ...]) -> None:
    assert list(values) == SCHEMA["$defs"][definition]["enum"]


CAPABILITY_FIXTURE = load("fixtures/core/capability-checks.json")


@pytest.mark.parametrize("case", CAPABILITY_FIXTURE["cases"], ids=lambda c: c["capability"])
def test_capability_checks(case: dict[str, Any]) -> None:
    caps = ProviderCapabilities.from_dict(CAPABILITY_FIXTURE["capabilities"])
    if case["ok"]:
        require_capability(caps, case["capability"])
        return
    with pytest.raises(ProviderError) as info:
        require_capability(caps, case["capability"])
    assert info.value.code == case["code"]
    if "messageContains" in case:
        assert case["messageContains"] in info.value.message
    validator("ProviderError").validate(info.value.to_dict())


FRAME_FIXTURE = load("fixtures/core/frame-validation.json")


@pytest.mark.parametrize("case", FRAME_FIXTURE["cases"])
def test_frame_validation(case: dict[str, Any]) -> None:
    f = case["format"]
    fmt = AudioFormat(f["encoding"], f["sampleRate"], f["channels"])
    if case["ok"]:
        validate_audio_payload(fmt, case["byteLength"], FRAME_FIXTURE["maxBytes"])
        return
    with pytest.raises(ProviderError) as info:
        validate_audio_payload(fmt, case["byteLength"], FRAME_FIXTURE["maxBytes"])
    assert info.value.code == case["code"]


G711 = load("fixtures/core/g711-mulaw.json")


def test_mulaw_decodes_all_codes() -> None:
    pcm = decode_mulaw(bytes(range(256)))
    assert list(struct.unpack("<256h", pcm)) == G711["decode"]


def test_mulaw_encodes_reference_samples() -> None:
    samples = [s for s, _ in G711["encode"]]
    encoded = encode_mulaw(struct.pack(f"<{len(samples)}h", *samples))
    assert list(encoded) == [u for _, u in G711["encode"]]


G711_ALAW = load("fixtures/core/g711-alaw.json")


def test_alaw_decodes_all_codes() -> None:
    pcm = decode_alaw(bytes(range(256)))
    assert list(struct.unpack("<256h", pcm)) == G711_ALAW["decode"]


def test_alaw_encodes_reference_samples() -> None:
    samples = [s for s, _ in G711_ALAW["encode"]]
    encoded = encode_alaw(struct.pack(f"<{len(samples)}h", *samples))
    assert list(encoded) == [a for _, a in G711_ALAW["encode"]]


def test_alaw_round_trips_every_code() -> None:
    assert encode_alaw(decode_alaw(bytes(range(256)))) == bytes(range(256))


def test_alaw_rejects_odd_length_pcm() -> None:
    with pytest.raises(ProviderError) as info:
        encode_alaw(b"\x00" * 3)
    assert info.value.code == "malformed_message"


@pytest.mark.parametrize("wire", SCHEMA_CASES["AudioFrame"]["valid"])
def test_frame_wire_round_trip(wire: dict[str, Any]) -> None:
    back = frame_to_wire(frame_from_wire(wire))
    validator("AudioFrame").validate(back)
    assert back["payloadBase64"] == wire["payloadBase64"]
