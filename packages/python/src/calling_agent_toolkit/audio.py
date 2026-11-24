# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
from __future__ import annotations

import base64
import struct
from collections.abc import Mapping
from typing import Any

from .errors import ProviderError
from .types import SUPPORTED_SAMPLE_RATES, AudioFormat, AudioFrame


def bytes_per_sample(fmt: AudioFormat) -> int:
    width = 2 if fmt.encoding == "pcm_s16le" else 1
    return width * fmt.channels


def validate_audio_payload(fmt: AudioFormat, byte_length: int, max_bytes: int, provider: str = "core") -> None:
    """Validate a payload size against its declared format. Does not inspect sample values."""
    if fmt.sample_rate not in SUPPORTED_SAMPLE_RATES:
        rates = ", ".join(str(r) for r in SUPPORTED_SAMPLE_RATES)
        raise ProviderError(
            code="unsupported_format",
            provider=provider,
            message=f"Sample rate {fmt.sample_rate} Hz is not supported; expected one of {rates}.",
        )
    if byte_length > max_bytes:
        raise ProviderError(
            code="buffer_overflow",
            provider=provider,
            message=f"Audio payload of {byte_length} bytes exceeds the {max_bytes}-byte limit.",
        )
    width = bytes_per_sample(fmt)
    if byte_length == 0 or byte_length % width != 0:
        raise ProviderError(
            code="malformed_message",
            provider=provider,
            message=f"Audio payload of {byte_length} bytes is not a whole number of {width}-byte {fmt.encoding} samples.",
        )


# ITU-T G.711 mu-law, bit-exact with CPython's audioop (contracts/fixtures/core/g711-mulaw.json).
# Encoding works on 14-bit samples (pcm16 >> 2, arithmetic shift), so small negatives round down.
_MULAW_BIAS = 0x84
_MULAW_CLIP_14 = 8159
_MULAW_SEGMENT_END_14 = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _decode_entry(byte: int) -> int:
    u = ~byte & 0xFF
    exponent = (u >> 4) & 0x07
    magnitude = ((((u & 0x0F) << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return -magnitude if u & 0x80 else magnitude


_MULAW_DECODE_TABLE = tuple(_decode_entry(i) for i in range(256))


def mulaw_to_linear(byte: int) -> int:
    return _MULAW_DECODE_TABLE[byte & 0xFF]


def linear_to_mulaw(sample: int) -> int:
    s = max(-32768, min(32767, int(sample))) >> 2
    mask = 0xFF
    if s < 0:
        s = -s
        mask = 0x7F
    s = min(s, _MULAW_CLIP_14) + (_MULAW_BIAS >> 2)
    for segment, end in enumerate(_MULAW_SEGMENT_END_14):
        if s <= end:
            return ((segment << 4) | ((s >> (segment + 1)) & 0x0F)) ^ mask
    return 0x7F ^ mask


def decode_mulaw(data: bytes) -> bytes:
    """mu-law bytes -> little-endian PCM16 bytes."""
    return struct.pack(f"<{len(data)}h", *(_MULAW_DECODE_TABLE[b] for b in data))


def encode_mulaw(pcm: bytes) -> bytes:
    """Little-endian PCM16 bytes -> mu-law bytes."""
    if len(pcm) % 2:
        raise ProviderError(code="malformed_message", provider="core", message="PCM16 input has an odd byte length.")
    return bytes(linear_to_mulaw(s) for s in struct.unpack(f"<{len(pcm) // 2}h", pcm))


# ITU-T G.711 A-law, bit-exact with CPython's audioop (contracts/fixtures/core/g711-alaw.json).
# Encoding works on 13-bit samples (pcm16 >> 3, arithmetic shift), so small negatives round down.
# Even bits are inverted on the wire (XOR 0x55); the sign bit is set for non-negative samples.
_ALAW_SEGMENT_END_13 = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _alaw_decode_entry(byte: int) -> int:
    a = byte ^ 0x55
    segment = (a >> 4) & 0x07
    mantissa = ((a & 0x0F) << 4) + (8 if segment == 0 else 0x108)
    magnitude = mantissa << (segment - 1) if segment > 1 else mantissa
    return magnitude if a & 0x80 else -magnitude


_ALAW_DECODE_TABLE = tuple(_alaw_decode_entry(i) for i in range(256))


def alaw_to_linear(byte: int) -> int:
    return _ALAW_DECODE_TABLE[byte & 0xFF]


def linear_to_alaw(sample: int) -> int:
    s = max(-32768, min(32767, int(sample))) >> 3
    mask = 0xD5
    if s < 0:
        s = -s - 1
        mask = 0x55
    for segment, end in enumerate(_ALAW_SEGMENT_END_13):
        if s <= end:
            return ((segment << 4) | ((s >> (1 if segment < 2 else segment)) & 0x0F)) ^ mask
    return 0x7F ^ mask


def decode_alaw(data: bytes) -> bytes:
    """A-law bytes -> little-endian PCM16 bytes."""
    return struct.pack(f"<{len(data)}h", *(_ALAW_DECODE_TABLE[b] for b in data))


def encode_alaw(pcm: bytes) -> bytes:
    """Little-endian PCM16 bytes -> A-law bytes."""
    if len(pcm) % 2:
        raise ProviderError(code="malformed_message", provider="core", message="PCM16 input has an odd byte length.")
    return bytes(linear_to_alaw(s) for s in struct.unpack(f"<{len(pcm) // 2}h", pcm))


def swap_bytes16(data: bytes) -> bytes:
    """Swap each 16-bit sample between big- and little-endian."""
    out = bytearray(data)
    out[0::2], out[1::2] = data[1::2], data[0::2]
    return bytes(out)


def frame_to_wire(frame: AudioFrame) -> dict[str, Any]:
    """Wire form of AudioFrame, matching core.schema.json."""
    return {
        "callId": frame.call_id,
        "sessionId": frame.session_id,
        "provider": frame.provider,
        "direction": frame.direction,
        "format": {
            "encoding": frame.format.encoding,
            "sampleRate": frame.format.sample_rate,
            "channels": frame.format.channels,
        },
        "payloadBase64": base64.b64encode(frame.payload).decode("ascii"),
        "timestampMs": frame.timestamp_ms,
        "timestampKind": frame.timestamp_kind,
        "sequenceNumber": frame.sequence_number,
        "chunk": frame.chunk,
        "isCompleteFrame": frame.is_complete_frame,
    }


def frame_from_wire(wire: Mapping[str, Any]) -> AudioFrame:
    fmt = wire["format"]
    return AudioFrame(
        call_id=wire["callId"],
        session_id=wire["sessionId"],
        provider=wire["provider"],
        direction=wire["direction"],
        format=AudioFormat(fmt["encoding"], fmt["sampleRate"], fmt["channels"]),
        payload=base64.b64decode(wire["payloadBase64"], validate=True),
        timestamp_ms=wire.get("timestampMs"),
        timestamp_kind=wire["timestampKind"],
        sequence_number=wire.get("sequenceNumber"),
        chunk=wire.get("chunk"),
        is_complete_frame=wire["isCompleteFrame"],
    )
