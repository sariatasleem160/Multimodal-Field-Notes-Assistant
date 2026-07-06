"""Generate sample audio via OpenAI TTS (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPTS = {
    "meeting_notes.mp3": (
        "Quick standup. Alex will send the revised timeline to the client by Thursday. "
        "We decided to postpone the release until QA signs off. "
        "Action item: book a follow-up call with the stakeholder next week."
    ),
    "voice_memo.mp3": (
        "Field note from site visit. The HVAC unit on roof three needs replacement filters. "
        "Order replacement filters before the next inspection. "
        "Pressure reading was slightly low but within tolerance."
    ),
}


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to generate audio samples.")

    client = OpenAI()
    for name, text in SCRIPTS.items():
        path = OUT / name
        print(f"Generating {name}...")
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        response.stream_to_file(str(path))
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
