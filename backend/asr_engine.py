#!/usr/bin/env python3
"""
ASR engine abstraction: picks an inference backend once, then exposes a single
transcribe_chunk(audio) used by pipeline.py -- so pipeline.py's VAD/diarization
logic stays identical regardless of platform.

Backends:
  - "mlx"          MLX Whisper (mlx_whisper) on the Apple-Silicon GPU (Metal).
                   Only importable on macOS + Apple Silicon; this is the setup
                   this project was originally built and benchmarked on.
  - "ctranslate2"  faster-whisper (CTranslate2). Cross-platform (Windows/
                   Linux/macOS): uses an NVIDIA GPU automatically if present
                   (float16), else falls back to CPU (int8) -- this is the
                   same faster-whisper + int8 CT2 model already validated
                   earlier in this project (persian_asr.py).

Selection is automatic ("auto"): tries mlx first (since it's faster on the
Mac this was developed on), falls back to ctranslate2 everywhere else. Force
one explicitly with the ASR_BACKEND env var ("mlx" | "ctranslate2") if the
auto-detection ever guesses wrong for your machine.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
MLX_MODEL_DIR = Path(os.environ.get("MLX_MODEL_DIR", str(_REPO_ROOT / "models" / "whisper-large-v3-persian-mlx-q8")))
CT2_MODEL_DIR = Path(os.environ.get("CT2_MODEL_DIR", str(_REPO_ROOT / "models" / "whisper-large-v3-persian-ct2-int8")))

_ct2_model = None
_active = None  # "mlx" | "ctranslate2", set on first use


def _try_mlx() -> bool:
    try:
        import mlx_whisper  # noqa: F401 -- availability probe only
    except Exception:
        return False
    return MLX_MODEL_DIR.exists()


def _try_ctranslate2() -> bool:
    global _ct2_model
    try:
        from faster_whisper import WhisperModel
        import ctranslate2
    except Exception:
        return False
    if not CT2_MODEL_DIR.exists():
        return False
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_count = 0
    device = "cuda" if cuda_count > 0 else "cpu"
    if device == "cuda":
        # Our CT2 model is already int8-quantized. "int8_float16" runs the
        # int8 weights directly on the GPU's INT8 Tensor Cores (fp16
        # accumulation) instead of dequantizing to float16 first -- faster
        # on Turing+ GPUs (e.g. T4) than plain "float16", and still supported
        # everywhere "float16" is (falls back automatically if unsupported).
        compute_type = os.environ.get("CT2_COMPUTE_TYPE", "int8_float16")
    else:
        compute_type = os.environ.get("CT2_COMPUTE_TYPE", "int8")
    _ct2_model = WhisperModel(str(CT2_MODEL_DIR), device=device, compute_type=compute_type)
    return True


def _select():
    global _active
    if _active is not None:
        return
    requested = os.environ.get("ASR_BACKEND", "auto").lower()
    if requested in ("mlx", "auto") and _try_mlx():
        _active = "mlx"
        return
    if requested in ("ctranslate2", "auto") and _try_ctranslate2():
        _active = "ctranslate2"
        return
    raise RuntimeError(
        "No usable ASR backend. Install mlx-whisper with a model at "
        f"{MLX_MODEL_DIR} (Apple Silicon), or faster-whisper with a model at "
        f"{CT2_MODEL_DIR} (Windows/Linux/other Mac)."
    )


def active_backend() -> str:
    _select()
    return _active


def _confidence(avg_logprob) -> "float | None":
    """avg_logprob (mean per-token log-probability, both backends already
    compute this for temperature-fallback / compression-ratio checks -- it was
    just never read past that) -> a 0..1 pseudo-confidence via exp(). Not a
    calibrated probability, but a real signal derived from the model's own
    output rather than a fabricated number."""
    if avg_logprob is None:
        return None
    try:
        import math
        return round(math.exp(avg_logprob), 4)
    except (OverflowError, ValueError):
        return None


def transcribe_chunk(audio, sample_rate: int = 16000, word_timestamps: bool = False) -> list:
    """Transcribe one audio chunk (float32 numpy array, `sample_rate` Hz).
    Returns [{"start": float, "end": float, "text": str, "confidence": float|None,
    "words": [...]|None}, ...], timestamps relative to the start of this chunk.

    With word_timestamps=True each segment carries a "words" list of
    {"start", "end", "text"} with timings the model actually aligned, via
    Whisper's cross-attention DTW. pipeline.py needs those to assign a speaker
    per word; without them it can only label a whole segment at once, so a
    segment spanning a speaker change gets one label for both people.

    It is off by default because the alignment pass costs real time (~10-20%)
    and only the diarizing path uses it."""
    _select()
    if _active == "mlx":
        import mlx_whisper
        r = mlx_whisper.transcribe(
            audio, path_or_hf_repo=str(MLX_MODEL_DIR), language="fa", task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
            no_speech_threshold=0.45, condition_on_previous_text=False,
            word_timestamps=word_timestamps, verbose=None)
        out = []
        for s in r.get("segments", []):
            # mlx_whisper returns dicts keyed "word"; faster-whisper uses .word
            # on an object. Normalize to "text" here so pipeline.py sees one shape.
            words = [{"start": round(w["start"], 3), "end": round(w["end"], 3),
                      "text": w.get("word", "")}
                     for w in (s.get("words") or [])] if word_timestamps else None
            out.append({"start": s["start"], "end": s["end"], "text": s["text"],
                        "confidence": _confidence(s.get("avg_logprob")),
                        "words": words or None})
        return out

    # ctranslate2 / faster-whisper -- pipeline.py already VAD-chunked the
    # audio, so vad_filter is off here to avoid re-segmenting a chunk that's
    # already speech-only.
    segments, _info = _ct2_model.transcribe(
        audio, language="fa", task="transcribe", beam_size=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold=2.4,
        no_speech_threshold=0.45, condition_on_previous_text=False,
        vad_filter=False, word_timestamps=word_timestamps)
    out = []
    for s in segments:
        words = [{"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word}
                 for w in (getattr(s, "words", None) or [])] if word_timestamps else None
        out.append({"start": s.start, "end": s.end, "text": s.text,
                    "confidence": _confidence(getattr(s, "avg_logprob", None)),
                    "words": words or None})
    return out
