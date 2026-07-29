#!/usr/bin/env python3
"""
Local chat / transcript-analysis LLM for the AI Analysis panel.

Backends:
  - "mlx"          Qwen3-4B-Instruct (MLX 4-bit) on the Apple-Silicon GPU.
                   Only importable on macOS + Apple Silicon.
  - "transformers" The same Qwen3-4B-Instruct model via HF `transformers`,
                   run on an NVIDIA GPU (float16) if available, else CPU.
                   Cross-platform: this is the path used on Windows/Linux.

Selection is automatic ("auto"): tries mlx first, falls back to transformers.
Force one explicitly with the CHAT_BACKEND env var ("mlx" | "transformers")
if the auto-detection ever guesses wrong for your machine.

Both backends expose the same public functions used by server.py:
stream_chat(messages, transcript) and make_title(transcript).
"""
import glob
import os
import threading

MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
HF_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen3.5-4B")
_SNAP_GLOB = "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/chat_template.jinja"

_active = None  # "mlx" | "transformers"
_MODEL = None
_TOK = None
_TMPL = None


def _try_mlx() -> bool:
    try:
        import mlx_lm  # noqa: F401 -- availability probe only
        return True
    except Exception:
        return False


def _select():
    global _active
    if _active is not None:
        return
    requested = os.environ.get("CHAT_BACKEND", "auto").lower()
    if requested in ("mlx", "auto") and _try_mlx():
        _active = "mlx"
    else:
        _active = "transformers"


def active_model_name() -> str:
    _select()
    return MLX_MODEL if _active == "mlx" else HF_MODEL


def _ensure():
    """Lazily load the active backend's model/tokenizer (and MLX chat
    template, if applicable)."""
    global _MODEL, _TOK, _TMPL
    _select()
    if _MODEL is not None:
        return _MODEL, _TOK, _TMPL
    if _active == "mlx":
        from mlx_lm import load
        _MODEL, _TOK = load(MLX_MODEL)
        hits = glob.glob(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), _SNAP_GLOB))
        _TMPL = open(hits[0]).read() if hits else None
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # CHAT_DEVICE forces cpu on a GPU box: the 4B fp16 weights are ~8GB, and
        # on a 16GB card that is more than half the board held for a panel that
        # is used on demand. CPU generation is much slower but leaves the GPU
        # entirely to ASR/diarization.
        want = os.environ.get("CHAT_DEVICE", "auto").lower()
        if want in ("cuda", "cpu"):
            device = want
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _TOK = AutoTokenizer.from_pretrained(HF_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype).to(device)
        _MODEL.eval()
    return _MODEL, _TOK, _TMPL


def unload() -> bool:
    """Drop the chat model and free its GPU memory. Returns True if something
    was actually unloaded.

    _ensure() caches the model for the process lifetime, so once the AI Analysis
    panel has been used once, ~8GB stays occupied for every subsequent
    transcription -- which is what pushes long files into CUDA OOM. Transcription
    calls this first; the next chat request reloads lazily (a few seconds).

    A generation already in flight keeps its own reference, so this is safe to
    call concurrently -- that memory just frees when the stream finishes."""
    global _MODEL, _TOK, _TMPL
    if _MODEL is None:
        return False
    _MODEL = _TOK = _TMPL = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def _system_prompt(transcript: str) -> str:
    if transcript:
        return (
            "You are a speech analytics AI assistant. You have access to the following "
            "transcript from a recorded conversation. Each line is prefixed with the "
            "speaker and the exact timestamp it was said, e.g. \"[S1 04:12]: ...\".\n\n"
            "TRANSCRIPT:\n" + transcript +
            "\n\nGuidelines:\n"
            "- Always respond in Persian (Farsi) regardless of the question's language\n"
            "- Keep answers brief and to the point\n"
            "- Reference specific speakers (S1, S2, S3, …) when relevant\n"
            "- When you state something the transcript says, cite EXACTLY where by copying "
            "that line's timestamp in brackets, e.g. [04:12] -- copy the digits verbatim from "
            "the transcript, do not estimate or reformat them. This lets the user jump to and "
            "verify that moment, so include one whenever you reference a specific claim, "
            "decision, or quote -- not for general summaries with no single source line"
        )
    return ("You are a helpful AI assistant specialized in speech transcription and audio "
            "analysis. Always respond in Persian (Farsi) briefly and to the point.")


def _build_messages(messages, transcript):
    msgs = [{"role": "system", "content": _system_prompt(transcript)}]
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def _context_window(model, tok) -> int:
    """Best-guess max sequence length the loaded model actually supports.
    transformers configs vary in which attribute carries this, and an unset
    tokenizer.model_max_length is a ~1e30 sentinel rather than a real number --
    both need guarding against, or the fallback below is silently never used."""
    ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if ctx and ctx < 1_000_000:
        return ctx
    mm = getattr(tok, "model_max_length", None)
    if mm and mm < 1_000_000:
        return mm
    return 32768


def _fit_to_context(tok, messages, transcript, max_new_tokens, max_ctx):
    """Shrink transcript then drop the oldest chat turns until the rendered
    prompt fits max_ctx - max_new_tokens. The full transcript is re-embedded
    in the system prompt on EVERY turn (see _system_prompt), so total tokens
    grow with both conversation length and transcript length -- on a long
    recording, a handful of follow-up questions is enough to exceed the
    model's context window. Whether that then errors or just produces
    garbage depends on the model, but either way the fix is the same: keep
    the rendered prompt under the ceiling before generating, rather than
    discovering the overflow from a failed generate() call."""
    def render(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    budget = max_ctx - max_new_tokens - 64
    msgs = _build_messages(messages, transcript)
    try:
        if len(render(msgs)) <= budget:
            return msgs
    except Exception:
        return msgs  # template/tokenizer surprised us -- let generate() surface it directly

    # 1) shrink the transcript first -- usually the largest single contributor,
    # and cutting it is less disruptive than dropping conversation turns
    t = transcript
    while t:
        t = t[: int(len(t) * 0.7)]
        msgs = _build_messages(messages, t)
        try:
            if len(render(msgs)) <= budget or not t:
                break
        except Exception:
            break

    # 2) still too long (a long chat history even with no transcript): drop
    # the oldest turns, always keeping at least the latest user message
    trimmed = list(messages)
    while len(trimmed) > 1:
        msgs = _build_messages(trimmed, t)
        try:
            if len(render(msgs)) <= budget:
                break
        except Exception:
            break
        trimmed.pop(0)
    return _build_messages(trimmed, t)


def stream_chat(messages, transcript="", max_tokens=1024, temperature=0.7):
    """Yield generated Persian text token-by-token, on whichever backend is active."""
    model, tok, tmpl = _ensure()

    if _active == "mlx":
        msgs = _build_messages(messages, transcript)
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        sampler = make_sampler(temp=temperature)
        for resp in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            if resp.text:
                yield resp.text
        return

    # transformers backend: generate in a background thread, stream via TextIteratorStreamer
    import torch
    from transformers import TextIteratorStreamer

    msgs = _fit_to_context(tok, messages, transcript, max_tokens, _context_window(model, tok))
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer,
                      do_sample=temperature > 0, temperature=max(temperature, 0.01))

    # generate() runs in its own thread so the streamer can be consumed here as
    # tokens arrive. Without the try/except, an exception in that thread (context
    # overflow, a transient CUDA OOM while ASR/diarization also hold the GPU) is
    # printed to stderr by Python's default thread excepthook and otherwise
    # vanishes -- crucially, streamer.end() is never called, so `for text in
    # streamer` below blocks forever waiting for a token that will never come.
    # That is the hang this project's users hit as "stops responding after a
    # few questions, have to start a new conversation": the request never
    # completes, so there is nothing for the client to time out on either.
    error: list = []

    def _run():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            error.append(e)
            streamer.end()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()
    if error:
        raise RuntimeError(f"generation failed: {error[0]}") from error[0]


def make_title(transcript: str) -> str:
    model, tok, tmpl = _ensure()
    msgs = [{"role": "user", "content":
             "Generate a concise 4-6 word Persian title for this transcript "
             "(no quotes, no trailing punctuation):\n\n" + transcript[:500]}]

    if _active == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        out = generate(model, tok, prompt=prompt, max_tokens=24,
                       sampler=make_sampler(temp=0.5), verbose=False).strip()
    else:
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=True, temperature=0.5)
        out = tok.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return out.splitlines()[0].strip('"“”') if out else "تحلیل جدید"
