# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
"""RNNoise noise suppression as an AudioProcessor (https://github.com/xiph/rnnoise).

The engine is the native ``librnnoise`` shared library, loaded at runtime through ``ctypes`` (standard library only),
so the core keeps zero dependencies and nothing is bundled. Point ``RNNOISE_LIBRARY`` (or ``library_path``) at the
library, or install it where the system loader finds ``rnnoise``.

RNNoise works on 480-sample frames at 48 kHz, so run the pipeline with ``target_sample_rate=48000, frame_ms=10``.
``AudioPipeline`` rejects a mismatch at construction. Use one processor per call: it keeps per-stream state.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import struct
from typing import Any

from ..audio_pipeline import ProcessorHealth, ProcessorKind
from ..errors import ProviderError

RNNOISE = "rnnoise"
RNNOISE_SAMPLE_RATE = 48000
RNNOISE_FRAME_SAMPLES = 480
_FRAME_BYTES = RNNOISE_FRAME_SAMPLES * 2
_UNPACK = struct.Struct(f"<{RNNOISE_FRAME_SAMPLES}h")


class RnnoiseProcessor:
    """Never raises for a missing or broken library: ``health()`` reports ``active=False`` with the reason, so a
    pipeline never claims suppression that is not running. ``process()`` then raises, and the pipeline bypasses it."""

    name = RNNOISE
    kind: ProcessorKind = "noise_suppression"
    sample_rate = RNNOISE_SAMPLE_RATE
    frame_samples = RNNOISE_FRAME_SAMPLES

    def __init__(self, library_path: str | os.PathLike[str] | None = None, *, library: Any = None) -> None:
        """``library`` injects an object exposing the librnnoise C functions (for tests); normally leave it unset."""
        self._lib: Any = None
        self._state: Any = None
        self._detail: str | None = None
        self._disposed = False
        self._voice_probability: float | None = None
        try:
            self._lib = library if library is not None else self._load_native(library_path)
            self._state = self._lib.rnnoise_create(None)
            if not self._state:
                raise RuntimeError("rnnoise_create returned no state")
        except Exception as error:  # a missing or broken engine must degrade, not crash the call
            self._lib = None
            self._state = None
            self._detail = f"RNNoise engine not loaded: {error}"
            return
        self._in = (ctypes.c_float * RNNOISE_FRAME_SAMPLES)()
        self._out = (ctypes.c_float * RNNOISE_FRAME_SAMPLES)()

    @staticmethod
    def _load_native(library_path: str | os.PathLike[str] | None) -> Any:
        path = os.fspath(library_path) if library_path is not None else os.environ.get("RNNOISE_LIBRARY") or ctypes.util.find_library("rnnoise")
        if not path:
            raise RuntimeError("librnnoise not found; install it or set RNNOISE_LIBRARY to its path")
        try:
            lib = ctypes.CDLL(path)
        except OSError as error:
            raise RuntimeError(f"could not load {path}: {error}") from error
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.restype = None
        lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        if hasattr(lib, "rnnoise_get_frame_size"):
            lib.rnnoise_get_frame_size.restype = ctypes.c_int
            size = lib.rnnoise_get_frame_size()
            if size != RNNOISE_FRAME_SAMPLES:
                raise RuntimeError(f"{path} uses {size}-sample frames, expected {RNNOISE_FRAME_SAMPLES}")
        return lib

    @property
    def voice_probability(self) -> float | None:
        """Voice-activity probability of the last processed frame, or None before the first."""
        return self._voice_probability

    def health(self) -> ProcessorHealth:
        if self._disposed:
            return ProcessorHealth(False, "RNNoise processor was disposed.")
        if self._lib is None:
            return ProcessorHealth(False, self._detail or "RNNoise engine not loaded.")
        return ProcessorHealth(True)

    def process(self, pcm: bytes) -> bytes:
        if self._disposed:
            raise ProviderError(code="session_closed", provider=RNNOISE, message="RNNoise processor was disposed.")
        if self._lib is None:
            raise ProviderError(code="invalid_configuration", provider=RNNOISE, message=self._detail or "RNNoise engine not loaded.")
        if len(pcm) != _FRAME_BYTES:
            raise ProviderError(
                code="malformed_message",
                provider=RNNOISE,
                message=f"RNNoise needs {RNNOISE_FRAME_SAMPLES} PCM16 samples ({_FRAME_BYTES} bytes) per frame, got {len(pcm)} bytes.",
            )
        self._in[:] = list(_UNPACK.unpack(pcm))
        self._voice_probability = float(self._lib.rnnoise_process_frame(self._state, self._out, self._in))
        return _UNPACK.pack(*(max(-32768, min(32767, math.floor(x + 0.5))) for x in self._out))

    def dispose(self) -> None:
        """Free the engine's state. The pipeline has no close hook, so call this when the call ends."""
        if self._disposed:
            return
        self._disposed = True
        if self._lib is not None:
            self._lib.rnnoise_destroy(self._state)
            self._lib = None
            self._state = None
