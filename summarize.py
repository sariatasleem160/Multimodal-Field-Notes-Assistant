"""Summarize transcripts into structured meeting notes."""

from __future__ import annotations

import json
import time
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from schemas import MeetingSummary

DEFAULT_SUMMARIZE_MODEL = "gpt-4o-mini"


def summarize_transcript(
    transcript: str,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_SUMMARIZE_MODEL,
    context_hint: str = "",
) -> tuple[MeetingSummary, float]:
    """Return structured summary and latency_ms."""
    client = client or OpenAI()

    system = (
        "You extract structured meeting notes from transcripts. "
        "Return JSON matching this shape: "
        '{"title": str|null, "summary": str, "key_topics": [str], '
        '"decisions": [str], "action_items": [{"task": str, "owner": str|null, '
        '"due_date": str|null, "priority": str|null}]}'
    )
    user = f"Transcript:\n\n{transcript}"
    if context_hint:
        user = f"Context: {context_hint}\n\n{user}"

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
        summary = MeetingSummary.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        summary = MeetingSummary(
            summary=raw if raw else "Could not parse structured summary.",
            key_topics=[],
            decisions=[],
            action_items=[],
        )

    return summary, latency_ms


def meeting_notes_to_markdown(summary: MeetingSummary, transcript: str = "") -> str:
    """Export meeting notes as Markdown (stretch goal)."""
    lines = [
        f"# {summary.title or 'Meeting Notes'}",
        "",
        "## Summary",
        summary.summary,
        "",
    ]
    if summary.key_topics:
        lines.extend(["## Key Topics", *[f"- {t}" for t in summary.key_topics], ""])
    if summary.decisions:
        lines.extend(["## Decisions", *[f"- {d}" for d in summary.decisions], ""])
    if summary.action_items:
        lines.append("## Action Items")
        lines.append("| Task | Owner | Due | Priority |")
        lines.append("|------|-------|-----|----------|")
        for item in summary.action_items:
            lines.append(
                f"| {item.task} | {item.owner or ''} | {item.due_date or ''} | {item.priority or ''} |"
            )
        lines.append("")
    if transcript:
        lines.extend(["## Full Transcript", "", transcript, ""])
    return "\n".join(lines)


def action_items_to_csv(summary: MeetingSummary) -> str:
    """Turn action items into CSV (stretch goal)."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["task", "owner", "due_date", "priority"])
    writer.writeheader()
    for item in summary.action_items:
        writer.writerow(
            {
                "task": item.task,
                "owner": item.owner or "",
                "due_date": item.due_date or "",
                "priority": item.priority or "",
            }
        )
    return buf.getvalue()
