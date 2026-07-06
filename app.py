"""Multimodal Field Notes Assistant — Streamlit UI."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from schemas import ExtractionType
from speak import text_to_speech
from summarize import action_items_to_csv, meeting_notes_to_markdown, summarize_transcript
from transcribe import transcribe_audio
from vision import analyze_image

load_dotenv()

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXTRACTION_LABELS = {
    ExtractionType.FREE_FORM: "Free-form Q&A",
    ExtractionType.RECEIPT: "Receipt",
    ExtractionType.WHITEBOARD: "Whiteboard",
    ExtractionType.CHART: "Chart / Screenshot",
    ExtractionType.EQUIPMENT: "Equipment / Field Photo",
}


def _client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set `OPENAI_API_KEY` in your environment or a `.env` file.")
        st.stop()
    return OpenAI()


def _save_temp_upload(uploaded, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return Path(tmp.name)


def image_tab() -> None:
    st.header("Image")
    st.caption("Upload a receipt, whiteboard, chart, screenshot, or field photo.")

    uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "webp", "gif"])
    extraction_key = st.selectbox(
        "Extraction mode",
        options=list(ExtractionType),
        format_func=lambda x: EXTRACTION_LABELS[x],
    )
    question = st.text_area(
        "Question or instruction",
        placeholder="e.g. What are the action items on this whiteboard?",
        height=80,
    )

    if "image_chat_history" not in st.session_state:
        st.session_state.image_chat_history = []

    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Analyze", type="primary", disabled=uploaded is None)
    with col2:
        if st.button("Clear follow-up chat"):
            st.session_state.image_chat_history = []

    if run and uploaded:
        path = _save_temp_upload(uploaded, Path(uploaded.name).suffix)
        with st.spinner("Running vision pipeline..."):
            result = analyze_image(
                path,
                question=question or "Describe and extract useful information.",
                extraction_type=extraction_key,
                client=_client(),
                chat_history=st.session_state.image_chat_history or None,
            )
        st.session_state.image_chat_history.append(
            {"role": "user", "content": question or "Analyze this image."}
        )
        st.session_state.image_chat_history.append(
            {"role": "assistant", "content": result.answer}
        )

        st.success(f"Done in **{result.latency_ms:.0f} ms** ({result.model})")
        st.subheader("Answer")
        st.write(result.answer)

        if result.structured_data:
            st.subheader("Structured data")
            st.json(result.structured_data)

        if not result.validation_ok:
            st.warning("Schema validation issues: " + "; ".join(result.validation_errors))

        with st.expander("Raw result"):
            st.json(result.model_dump())

    if st.session_state.image_chat_history:
        st.subheader("Follow-up chat")
        follow_up = st.text_input("Ask a follow-up about the same image")
        if st.button("Send follow-up", disabled=uploaded is None) and follow_up and uploaded:
            path = _save_temp_upload(uploaded, Path(uploaded.name).suffix)
            with st.spinner("Thinking..."):
                result = analyze_image(
                    path,
                    question=follow_up,
                    extraction_type=ExtractionType.FREE_FORM,
                    client=_client(),
                    chat_history=st.session_state.image_chat_history,
                )
            st.session_state.image_chat_history.append({"role": "user", "content": follow_up})
            st.session_state.image_chat_history.append({"role": "assistant", "content": result.answer})
            st.info(result.answer)


def audio_tab() -> None:
    st.header("Audio")
    st.caption("Upload a voice memo or meeting recording.")

    uploaded = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "webm", "mp4"])
    speaker_labels = st.checkbox("Approximate speaker labels (best-effort)", value=False)
    gen_speech = st.checkbox("Generate spoken summary (TTS)", value=False)

    if st.button("Process audio", type="primary", disabled=uploaded is None) and uploaded:
        path = _save_temp_upload(uploaded, Path(uploaded.name).suffix)
        client = _client()

        with st.spinner("Transcribing..."):
            transcript, segments, t_lat = transcribe_audio(
                path, client=client, with_speaker_labels=speaker_labels
            )

        with st.spinner("Summarizing..."):
            summary, s_lat = summarize_transcript(transcript, client=client)

        tts_path = None
        tts_lat = 0.0
        if gen_speech:
            with st.spinner("Generating speech..."):
                out = RESULTS_DIR / f"spoken_{Path(uploaded.name).stem}.mp3"
                tts_path, tts_lat = text_to_speech(summary.summary, out, client=client)

        total = t_lat + s_lat + tts_lat
        st.success(
            f"Pipeline complete in **{total:.0f} ms** "
            f"(transcribe {t_lat:.0f} + summarize {s_lat:.0f}"
            + (f" + TTS {tts_lat:.0f}" if gen_speech else "")
            + ")"
        )

        st.subheader("Transcript")
        st.write(transcript)
        if segments:
            with st.expander("Segments"):
                for seg in segments:
                    label = f"[{seg.speaker}] " if seg.speaker else ""
                    st.caption(f"{seg.start:.1f}s–{seg.end:.1f}s")
                    st.write(f"{label}{seg.text}")

        st.subheader("Summary")
        st.write(summary.summary)

        if summary.decisions:
            st.subheader("Decisions")
            for d in summary.decisions:
                st.write(f"- {d}")

        if summary.action_items:
            st.subheader("Action items")
            for item in summary.action_items:
                meta = []
                if item.owner:
                    meta.append(f"owner: {item.owner}")
                if item.due_date:
                    meta.append(f"due: {item.due_date}")
                suffix = f" ({', '.join(meta)})" if meta else ""
                st.write(f"- {item.task}{suffix}")

        md = meeting_notes_to_markdown(summary, transcript)
        csv_data = action_items_to_csv(summary)

        st.download_button("Export Markdown", md, file_name="meeting_notes.md", mime="text/markdown")
        if summary.action_items:
            st.download_button("Export action items CSV", csv_data, file_name="action_items.csv", mime="text/csv")

        if tts_path and tts_path.exists():
            st.subheader("Spoken summary")
            st.audio(str(tts_path))


def main() -> None:
    st.set_page_config(page_title="Field Notes Assistant", page_icon="📋", layout="wide")
    st.title("Multimodal Field Notes Assistant")
    st.markdown(
        "Turn **photos** and **voice memos** into structured work artifacts — "
        "receipts, whiteboard notes, meeting summaries, and action items."
    )

    tab_image, tab_audio = st.tabs(["Image", "Audio"])
    with tab_image:
        image_tab()
    with tab_audio:
        audio_tab()

    with st.sidebar:
        st.header("About")
        st.markdown(
            """
**Image flow:** image → vision model → JSON / answer → validation

**Audio flow:** audio → STT → transcript → LLM summary → optional TTS

**Limitations:** handwriting, noisy audio, dense charts, long recordings, and unclear photos reduce accuracy.
            """
        )


if __name__ == "__main__":
    main()
