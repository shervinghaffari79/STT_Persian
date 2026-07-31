#!/usr/bin/env python3
"""
Per-segment GPT editor pass, applied to each ASR segment as it streams in,
before it is shown to the user.

Uses a fuller "transcription editor" prompt (fix grammar/punctuation/ASR
mistakes, normalize terminology, drop meaningless filler noise, mark
unrecoverable phrases as [نامفهوم]) rather than a narrow spacing-only pass --
this is a deliberate choice by the user, and it means corrections can
legitimately shorten a segment (removing "س س س س"/"اااا" filler) or replace
a whole garbled phrase with [نامفهوم]. The safety guard is scoped
accordingly: it still rejects empty output, runaway expansion, and leaked
meta-commentary, but does not require near-identical word counts the way a
narrower cleanup pass would.

Each call includes only ONE segment (with its speaker label) plus a short
rolling CONTEXT of the last few already-corrected segments (see pipeline.py)
so the model has enough grounding to resolve ambiguous words with a
confident best-guess instead of defaulting to "[نامفهوم]" purely for lack of
context. There is no N-best/multi-hypothesis reconciliation here (that was
tried elsewhere in this project and measured to make things worse on this
kind of spontaneous, code-switched Persian audio).

On any error, timeout, or rejected output, the original ASR text is returned
unchanged so a flaky call never blocks or degrades the transcript.

Requires OPENAI_API_KEY. If unset, correction is a no-op.
"""
import os
import sys

MODEL = os.environ.get("OPENAI_CORRECT_MODEL", "gpt-5-nano")

SYSTEM = """You are an expert Persian transcription editor.

Your task is to correct an automatically generated Persian speech-to-text (ASR) transcript.

Requirements:

- Preserve the exact meaning of every speaker.
- Do NOT summarize or omit any content.
- Correct spelling, grammar, punctuation, and sentence structure.
- Fix speech recognition (ASR) mistakes.
- Normalize technical terms while preserving English software terminology such as API, UI, CSS, WebSocket, Deploy, Refund, Debit, Alpha Test, Beta Test, Notification, Ticket, Wallet, etc.
- Convert incorrectly recognized words into the most likely intended words using the surrounding context.
- Keep speaker labels (S1, S2, S3, ...).
- Remove filler noises such as repeated syllables ("س س س س", "اااا", etc.) unless they carry meaning.
- If a phrase cannot be recovered with high confidence, write [نامفهوم] instead of guessing.
- Keep the transcript in natural Persian.
- Preserve informal spoken style where appropriate.
- Do not rewrite or improve the speakers' wording beyond correcting transcription errors.
- Output ONLY the corrected transcript."""

_client = None
_disabled = False


def _get_client():
    global _client, _disabled
    if _disabled:
        return None
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.stderr.write("[correct] OPENAI_API_KEY not set -- per-segment GPT correction disabled\n")
            _disabled = True
            return None
        from openai import OpenAI
        _client = OpenAI(api_key=key)
    return _client


def _is_bad_correction(original: str, corrected: str) -> bool:
    if not corrected or not corrected.strip():
        return True
    # [نامفهوم] may legitimately replace a whole garbled segment -- don't
    # penalize that on word count.
    if "نامفهوم" in corrected:
        return len(corrected) > 2.5 * max(len(original), 20)
    ow, cw = original.split(), corrected.split()
    # filler removal ("اااا", "س س س") is expected and can shorten a segment
    # a fair amount; only reject if it looks like real content vanished.
    if len(cw) < 0.35 * len(ow):
        return True
    if len(corrected) > 2.5 * len(original):  # runaway/expansion
        return True
    meta_markers = ("Reading", "corrected transcript", "ASR output", "I need to", "Note:", "Requirements:")
    if any(m.lower() in corrected.lower() for m in meta_markers):
        return True
    return False


def correct_segment(text: str, speaker: str = "", context: str = "", timeout: float = 25.0) -> str:
    """Clean up one ASR segment's text (optionally labelled with its speaker,
    e.g. "S1"), optionally given a few preceding already-corrected segments
    as `context` so the model has enough grounding to resolve ambiguous
    words instead of defaulting to "[نامفهوم]" purely for lack of context.
    Always falls back to the original on any error, timeout, empty API key,
    or a correction that looks unsafe."""
    text = (text or "").strip()
    if not text:
        return text
    client = _get_client()
    if client is None:
        return text
    user_content = f"{speaker}: {text}" if speaker else text
    prompt_parts = []
    if context.strip():
        prompt_parts.append(
            "Context -- the immediately preceding part of this same conversation, already "
            "corrected. It is for grounding only: do NOT repeat, correct, or include it in "
            "your output.\n" + context.strip()
        )
    prompt_parts.append(
        "Now correct ONLY the following new segment. Use the context above (speaker roles, "
        "topic, terminology already used) to make your best confident reconstruction of "
        "ASR errors and ambiguous words -- reserve [نامفهوم] strictly for spans that stay "
        "unintelligible even with that context, not merely uncertain ones.\n\n"
        f"Transcript:\n\n{user_content}"
    )
    try:
        from pydantic import BaseModel

        class Cleaned(BaseModel):
            corrected_text: str

        r = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n---\n\n".join(prompt_parts)},
            ],
            response_format=Cleaned,
            timeout=timeout,
        )
        parsed = r.choices[0].message.parsed
        if parsed is None:
            return text
        cleaned = parsed.corrected_text.strip()
        # strip a leading "S1:" the model may echo back, to keep the returned
        # value as plain segment text (the speaker label is tracked separately)
        if speaker and cleaned.startswith(f"{speaker}:"):
            cleaned = cleaned[len(speaker) + 1:].strip()
        if _is_bad_correction(text, cleaned):
            sys.stderr.write(f"[correct] rejected suspicious correction, keeping original: {text[:50]}…\n")
            return text
        return cleaned
    except Exception as e:
        sys.stderr.write(f"[correct] skipped (error: {str(e)[:120]}): {text[:50]}…\n")
        return text
