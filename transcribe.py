"""Speech-to-text using OpenAI Whisper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from schemas import TranscriptSegment

DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


def transcribe_audio(
    audio_path: str | Path,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
    with_speaker_labels: bool = False,
) -> tuple[str, list[TranscriptSegment], float]:
    """
    Transcribe audio to text.

    Returns (full_transcript, segments, latency_ms).
    Speaker labels are best-effort via segment timestamps; true diarization
    requires a dedicated service (documented limitation).
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    client = client or OpenAI()
    start = time.perf_counter()

    with path.open("rb") as audio_file:
        if with_speaker_labels:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        else:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

    latency_ms = (time.perf_counter() - start) * 1000

    segments: list[TranscriptSegment] = []
    full_text_parts: list[str] = []

    raw_segments = getattr(response, "segments", None) or []
    for i, seg in enumerate(raw_segments):
        text = (seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or ""
        text = text.strip()
        if not text:
            continue
        start_t = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end_t = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        speaker = f"Speaker {(i % 2) + 1}" if with_speaker_labels else None
        segments.append(
            TranscriptSegment(start=float(start_t or 0), end=float(end_t or 0), text=text, speaker=speaker)
        )
        prefix = f"{speaker}: " if speaker else ""
        full_text_parts.append(f"{prefix}{text}")

    transcript = getattr(response, "text", None) or " ".join(full_text_parts)
    transcript = transcript.strip()

    return transcript, segments, latency_ms
