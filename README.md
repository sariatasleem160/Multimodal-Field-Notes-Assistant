# Multimodal Field Notes Assistant

Turn **photos** and **voice memos** into structured work artifacts — receipts, whiteboard notes, meeting summaries, and action items.

## What it does

| Tab | Input | Output |
|-----|-------|--------|
| **Image** | Receipt, whiteboard, chart, screenshot, field photo | Answer + optional JSON (validated schema) |
| **Audio** | Voice memo or meeting recording | Transcript, summary, decisions, action items, optional TTS |

## Why it matters

Real work is not only typed text. People photograph whiteboards, scan receipts, record meetings, and leave voice notes. This project shows how to chain **vision**, **speech-to-text**, **LLM summarization**, and **text-to-speech** into a usable pipeline with latency metrics.

## File structure

```
03-multimodal-field-notes/
├── samples/
│   ├── images/          # 5+ example images
│   └── audio/           # 2+ example recordings
├── vision.py            # Image → vision model → JSON
├── transcribe.py        # Audio → Whisper transcript
├── summarize.py         # Transcript → structured notes
├── speak.py             # Text → speech
├── schemas.py           # Pydantic models
├── app.py               # Streamlit UI (two tabs)
├── evaluate.py          # Batch eval + latency
├── results/
│   ├── image_outputs.jsonl
│   ├── transcript_outputs.jsonl
│   └── latency_summary.json
├── scripts/
│   ├── generate_sample_images.py
│   └── generate_sample_audio.py
├── requirements.txt
└── README.md
```

## Data flows

### Image flow

```
image → vision model + question/schema → JSON or answer → Pydantic validation → UI
```

### Audio flow

```
audio → Whisper (STT) → transcript → LLM summary → action items → optional TTS
```

## Setup

```bash
cd 03-multimodal-field-notes
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
copy .env.example .env          # add OPENAI_API_KEY
```

### Generate sample files

```bash
python scripts/generate_sample_images.py    # no API key needed
python scripts/generate_sample_audio.py     # requires OPENAI_API_KEY
```

## Run the app

```bash
streamlit run app.py
```

Open the **Image** tab to upload a photo and extract structured data. Open the **Audio** tab for transcript + summary + export.

## Run evaluation

```bash
python evaluate.py
```

Writes:

- `results/image_outputs.jsonl` — one JSON object per image
- `results/transcript_outputs.jsonl` — transcript + summary per audio file
- `results/latency_summary.json` — mean/min/max latency by flow

Action-item accuracy is scored against a small expected-answer set in `evaluate.py` using token overlap.

## Portfolio metrics

| Metric | Target |
|--------|--------|
| Image examples | ≥ 5 with screenshots in README or `results/` |
| Audio examples | ≥ 2 with transcript + summary |
| Action-item accuracy | Manual expected set in `evaluate.py` |
| Latency | Reported in UI and `latency_summary.json` |
| Documented limitation | See below |

## Stretch goals (included)

- **Follow-up chat** with the same image (Image tab)
- **Export meeting notes** to Markdown
- **Action items** to CSV
- **Approximate speaker labels** (segment-based; not true diarization)

## Known limitations

1. **Handwriting** — messy or cursive writing is often misread.
2. **Noisy audio** — background noise degrades Whisper accuracy.
3. **Dense charts** — small labels and 3D plots are easy to misinterpret.
4. **Long recordings** — very long files increase cost and latency; chunking not implemented.
5. **Unclear photos** — blur, glare, and low light hurt vision extraction.
6. **Speaker labels** — alternating pseudo-speakers only; real diarization needs a dedicated model.

## Models used

| Step | Default model |
|------|----------------|
| Vision | `gpt-4o` |
| Transcription | `whisper-1` |
| Summarization | `gpt-4o-mini` |
| TTS | `tts-1` |

## Environment

| Variable | Required |
|----------|----------|
| `OPENAI_API_KEY` | Yes |

## License

MIT — portfolio / learning use.
