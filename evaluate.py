"""Batch evaluation: latency, image outputs, transcript outputs, action-item accuracy."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from schemas import EvaluationCase, ExtractionType, LatencyRecord
from summarize import summarize_transcript
from transcribe import transcribe_audio
from vision import analyze_image

load_dotenv()

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMAGE_QUESTIONS = {
    "receipt": ("Extract all line items and totals.", ExtractionType.RECEIPT),
    "whiteboard": ("List bullet points and action items.", ExtractionType.WHITEBOARD),
    "chart": ("Describe the chart and key data points.", ExtractionType.CHART),
    "equipment": ("Identify equipment and any visible issues.", ExtractionType.EQUIPMENT),
    "screenshot": ("Summarize the main content.", ExtractionType.FREE_FORM),
}

EVAL_CASES = [
    EvaluationCase(
        audio_file="meeting_notes",
        expected_action_items=[
            {"task": "send revised timeline to client", "owner": "Alex"},
            {"task": "book follow-up call", "owner": None},
        ],
    ),
    EvaluationCase(
        audio_file="voice_memo",
        expected_action_items=[
            {"task": "order replacement filters", "owner": None},
        ],
    ),
]


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def action_item_accuracy(
    predicted: list[dict],
    expected: list[dict],
    threshold: float = 0.4,
) -> dict:
    """Match predicted action items to expected via token overlap."""
    matched = 0
    details = []
    for exp in expected:
        best_score = 0.0
        best_pred = None
        for pred in predicted:
            score = _token_overlap(exp["task"], pred.get("task", ""))
            if score > best_score:
                best_score = score
                best_pred = pred.get("task", "")
        hit = best_score >= threshold
        if hit:
            matched += 1
        details.append(
            {
                "expected": exp["task"],
                "best_match": best_pred,
                "score": round(best_score, 3),
                "hit": hit,
            }
        )
    total = len(expected) or 1
    return {
        "accuracy": round(matched / total, 3),
        "matched": matched,
        "total": len(expected),
        "details": details,
    }


def run_image_eval(client: OpenAI) -> tuple[list[dict], list[LatencyRecord]]:
    image_dir = SAMPLES / "images"
    outputs: list[dict] = []
    latencies: list[LatencyRecord] = []

    if not image_dir.exists():
        print(f"No image samples at {image_dir}")
        return outputs, latencies

    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            continue
        stem = path.stem.lower()
        question, ext_type = IMAGE_QUESTIONS.get(
            stem.split("_")[0],
            ("Describe this image.", ExtractionType.FREE_FORM),
        )
        print(f"  Image: {path.name}")
        try:
            result = analyze_image(path, question, ext_type, client=client)
            outputs.append({"file": path.name, **result.model_dump()})
            latencies.append(
                LatencyRecord(
                    flow="image",
                    example_id=path.name,
                    step="vision",
                    latency_ms=result.latency_ms,
                    model=result.model,
                )
            )
        except Exception as exc:
            outputs.append({"file": path.name, "error": str(exc)})
            print(f"    Error: {exc}")

    return outputs, latencies


def _resolve_audio(stem: str, audio_dir: Path) -> Path | None:
    for ext in (".mp3", ".wav", ".m4a", ".webm"):
        path = audio_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def run_audio_eval(client: OpenAI) -> tuple[list[dict], list[LatencyRecord], list[dict]]:
    audio_dir = SAMPLES / "audio"
    outputs: list[dict] = []
    latencies: list[LatencyRecord] = []
    accuracy_reports: list[dict] = []

    if not audio_dir.exists():
        print(f"No audio samples at {audio_dir}")
        return outputs, latencies, accuracy_reports

    for case in EVAL_CASES:
        path = _resolve_audio(case.audio_file, audio_dir)
        if path is None:
            print(f"  Skip missing: {case.audio_file} (.mp3/.wav)")
            continue
        print(f"  Audio: {path.name}")
        try:
            t0 = time.perf_counter()
            transcript, segments, t_lat = transcribe_audio(path, client=client)
            summary, s_lat = summarize_transcript(transcript, client=client)
            total = (time.perf_counter() - t0) * 1000

            pred_items = [item.model_dump() for item in summary.action_items]
            exp_items = [e.model_dump() for e in case.expected_action_items]
            acc = action_item_accuracy(pred_items, exp_items)

            record = {
                "file": path.name,
                "transcript": transcript,
                "summary": summary.model_dump(),
                "action_item_accuracy": acc,
                "transcribe_latency_ms": t_lat,
                "summarize_latency_ms": s_lat,
                "total_latency_ms": total,
            }
            outputs.append(record)
            accuracy_reports.append({"file": case.audio_file, **acc})

            latencies.extend(
                [
                    LatencyRecord(
                        flow="audio",
                        example_id=case.audio_file,
                        step="transcribe",
                        latency_ms=t_lat,
                    ),
                    LatencyRecord(
                        flow="audio",
                        example_id=case.audio_file,
                        step="summarize",
                        latency_ms=s_lat,
                    ),
                    LatencyRecord(
                        flow="audio",
                        example_id=case.audio_file,
                        step="total",
                        latency_ms=total,
                    ),
                ]
            )
        except Exception as exc:
            outputs.append({"file": case.audio_file, "error": str(exc)})
            print(f"    Error: {exc}")

    return outputs, latencies, accuracy_reports


def summarize_latencies(records: list[LatencyRecord]) -> dict:
    by_flow: dict[str, list[float]] = {}
    for r in records:
        by_flow.setdefault(r.flow, []).append(r.latency_ms)

    summary = {}
    for flow, values in by_flow.items():
        summary[flow] = {
            "count": len(values),
            "mean_ms": round(sum(values) / len(values), 1),
            "min_ms": round(min(values), 1),
            "max_ms": round(max(values), 1),
        }
    return summary


def main() -> None:
    client = OpenAI()
    print("Running image evaluation...")
    image_outputs, latencies = run_image_eval(client)

    print("Running audio evaluation...")
    audio_outputs, audio_latencies, accuracy = run_audio_eval(client)
    latencies.extend(audio_latencies)

    image_path = RESULTS / "image_outputs.jsonl"
    with image_path.open("w", encoding="utf-8") as f:
        for row in image_outputs:
            f.write(json.dumps(row) + "\n")

    transcript_path = RESULTS / "transcript_outputs.jsonl"
    with transcript_path.open("w", encoding="utf-8") as f:
        for row in audio_outputs:
            f.write(json.dumps(row) + "\n")

    latency_summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by_flow": summarize_latencies(latencies),
        "records": [r.model_dump() for r in latencies],
        "action_item_accuracy": accuracy,
    }
    latency_path = RESULTS / "latency_summary.json"
    latency_path.write_text(json.dumps(latency_summary, indent=2), encoding="utf-8")

    print(f"\nWrote {image_path}")
    print(f"Wrote {transcript_path}")
    print(f"Wrote {latency_path}")
    if accuracy:
        for a in accuracy:
            print(f"  Action-item accuracy ({a['file']}): {a['accuracy']:.0%}")


if __name__ == "__main__":
    main()
