"""Pydantic schemas for structured multimodal outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractionType(str, Enum):
    FREE_FORM = "free_form"
    RECEIPT = "receipt"
    WHITEBOARD = "whiteboard"
    CHART = "chart"
    EQUIPMENT = "equipment"


class ReceiptData(BaseModel):
    merchant: Optional[str] = None
    date: Optional[str] = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    total: Optional[str] = None
    currency: Optional[str] = None


class WhiteboardData(BaseModel):
    title: Optional[str] = None
    bullet_points: list[str] = Field(default_factory=list)
    diagrams_described: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class ChartData(BaseModel):
    chart_type: Optional[str] = None
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    data_points: list[dict[str, Any]] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)


class EquipmentData(BaseModel):
    equipment_type: Optional[str] = None
    model_or_label: Optional[str] = None
    visible_condition: Optional[str] = None
    readings: list[dict[str, str]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


EXTRACTION_SCHEMAS: dict[ExtractionType, type[BaseModel]] = {
    ExtractionType.RECEIPT: ReceiptData,
    ExtractionType.WHITEBOARD: WhiteboardData,
    ExtractionType.CHART: ChartData,
    ExtractionType.EQUIPMENT: EquipmentData,
}


class ImageResult(BaseModel):
    question: str
    extraction_type: str
    answer: str
    structured_data: Optional[dict[str, Any]] = None
    validation_ok: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""


class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None


class MeetingSummary(BaseModel):
    title: Optional[str] = None
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class AudioPipelineResult(BaseModel):
    transcript: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    summary: MeetingSummary
    spoken_summary_path: Optional[str] = None
    transcribe_latency_ms: float = 0.0
    summarize_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    model_transcribe: str = ""
    model_summarize: str = ""


class LatencyRecord(BaseModel):
    flow: str
    example_id: str
    step: str
    latency_ms: float
    model: str = ""


class ExpectedActionItem(BaseModel):
    """Ground-truth action item for evaluation."""

    task: str
    owner: Optional[str] = None


class EvaluationCase(BaseModel):
    audio_file: str
    expected_action_items: list[ExpectedActionItem] = Field(default_factory=list)
