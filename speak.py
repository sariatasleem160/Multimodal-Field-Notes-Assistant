"""Text-to-speech for spoken summaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_VOICE = "alloy"


def text_to_speech(
    text: str,
    output_path: str | Path,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
) -> tuple[Path, float]:
    """Synthesize speech and return (output_path, latency_ms)."""
    client = client or OpenAI()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    response = client.audio.speech.create(model=model, voice=voice, input=text[:4096])
    response.stream_to_file(str(out))
    latency_ms = (time.perf_counter() - start) * 1000

    return out, latency_ms
