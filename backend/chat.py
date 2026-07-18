#!/usr/bin/env python3
"""
Local chat / transcript-analysis LLM for the AI Analysis panel.

Runs Qwen3-4B-Instruct (MLX 4-bit) on the Apple-Silicon GPU — the same local
model used for the ASR error-correction experiments — replacing the cloud
OpenRouter call. Streams tokens for a responsive UI.
"""
import glob
import os

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
_SNAP_GLOB = "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/chat_template.jinja"

_MODEL = None
_TOK = None
_TMPL = None


def _ensure():
    global _MODEL, _TOK, _TMPL
    if _MODEL is None:
        from mlx_lm import load
        _MODEL, _TOK = load(MODEL)
        hits = glob.glob(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), _SNAP_GLOB))
        _TMPL = open(hits[0]).read() if hits else None
    return _MODEL, _TOK, _TMPL


def _system_prompt(transcript: str) -> str:
    if transcript:
        return (
            "You are a speech analytics AI assistant. You have access to the following "
            "transcript from a recorded conversation.\n\nTRANSCRIPT:\n" + transcript +
            "\n\nGuidelines:\n"
            "- Always respond in Persian (Farsi) regardless of the question's language\n"
            "- Keep answers brief and to the point\n"
            "- Reference specific speakers (S1, S2, S3, …) when relevant\n"
            "- Include timestamps only when directly useful"
        )
    return ("You are a helpful AI assistant specialized in speech transcription and audio "
            "analysis. Always respond in Persian (Farsi) briefly and to the point.")


def _prompt(messages, transcript):
    model, tok, tmpl = _ensure()
    msgs = [{"role": "system", "content": _system_prompt(transcript)}]
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)


def stream_chat(messages, transcript="", max_tokens=1024, temperature=0.7):
    """Yield generated Persian text token-by-token."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok, _ = _ensure()
    prompt = _prompt(messages, transcript)
    sampler = make_sampler(temp=temperature)
    for resp in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
        if resp.text:
            yield resp.text


def make_title(transcript: str) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model, tok, tmpl = _ensure()
    msgs = [{"role": "user", "content":
             "Generate a concise 4-6 word Persian title for this transcript "
             "(no quotes, no trailing punctuation):\n\n" + transcript[:500]}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
    out = generate(model, tok, prompt=prompt, max_tokens=24,
                   sampler=make_sampler(temp=0.5), verbose=False).strip()
    return out.splitlines()[0].strip('"“”') if out else "تحلیل جدید"
