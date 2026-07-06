"""Vision pipeline: image -> model with question/schema -> JSON or answer."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from schemas import EXTRACTION_SCHEMAS, ExtractionType, ImageResult

DEFAULT_VISION_MODEL = "gpt-4o"


def _encode_image(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return mime, data


def _schema_prompt(extraction_type: ExtractionType) -> str:
    if extraction_type == ExtractionType.FREE_FORM:
        return (
            "Answer the user's question about this image clearly and concisely. "
            "Return JSON: {\"answer\": \"...\"}"
        )
    schema_cls = EXTRACTION_SCHEMAS[extraction_type]
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        f"Extract structured information from this image as type '{extraction_type.value}'. "
        f"Return JSON with keys \"answer\" (brief human summary) and \"structured_data\" "
        f"(object matching this schema):\n{schema_json}"
    )


def analyze_image(
    image_path: str | Path,
    question: str,
    extraction_type: ExtractionType = ExtractionType.FREE_FORM,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_VISION_MODEL,
    chat_history: Optional[list[dict[str, Any]]] = None,
) -> ImageResult:
    """
    Send an image to a vision-capable model and return a validated result.

  `chat_history` enables follow-up questions: prior turns as OpenAI message dicts.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    client = client or OpenAI()
    mime, b64 = _encode_image(path)
    image_url = f"data:{mime};base64,{b64}"

    system = (
        "You are a field-notes assistant that reads photos of receipts, whiteboards, "
        "charts, screenshots, and equipment. Always respond with valid JSON only."
    )
    user_text = question.strip() or "Describe what you see and extract useful details."
    if extraction_type != ExtractionType.FREE_FORM:
        user_text = f"{user_text}\n\n{_schema_prompt(extraction_type)}"
    else:
        user_text = f"{user_text}\n\n{_schema_prompt(ExtractionType.FREE_FORM)}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if chat_history:
        messages.extend(chat_history)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1500,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"answer": raw, "structured_data": None}

    answer = str(payload.get("answer", payload.get("response", raw)))
    structured = payload.get("structured_data")

    validation_ok = True
    validation_errors: list[str] = []
    if extraction_type != ExtractionType.FREE_FORM and structured is not None:
        schema_cls = EXTRACTION_SCHEMAS[extraction_type]
        try:
            validated = schema_cls.model_validate(structured)
            structured = validated.model_dump()
        except ValidationError as exc:
            validation_ok = False
            validation_errors = [e["msg"] for e in exc.errors()]

    return ImageResult(
        question=question,
        extraction_type=extraction_type.value,
        answer=answer,
        structured_data=structured,
        validation_ok=validation_ok,
        validation_errors=validation_errors,
        latency_ms=latency_ms,
        model=model,
    )
