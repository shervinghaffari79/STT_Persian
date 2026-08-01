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
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_MODEL_DIR = Path(os.environ.get("MLX_MODEL_DIR", str(_REPO_ROOT / "models" / "whisper-large-v3-persian-mlx-q8")))
CT2_MODEL_DIR = Path(os.environ.get("CT2_MODEL_DIR", str(_REPO_ROOT / "models" / "whisper-large-v3-persian-ct2-int8")))

# Beam search width. 5 is Whisper's default and what this model was evaluated
# at; decode cost scales roughly with it, so WHISPER_BEAM_SIZE=1 (greedy) is
# the largest single ASR speedup available here, at some accuracy cost. Left at
# 5 so speed is never traded for WER without someone choosing to.
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

_ct2_model = None
_active = None  # "mlx" | "ctranslate2", set on first use
_SELECT_LOCK = threading.Lock()
# Diagnostic snapshot of what _select() actually decided and why -- see
# backend_info(). Populated as a side effect of _try_mlx()/_try_ctranslate2()
# so a slow deploy can be checked (e.g. via /api/asr-status) without SSH+profiling.
_device_info: dict = {}


def _try_mlx() -> bool:
    try:
        import mlx_whisper  # noqa: F401 -- availability probe only
    except Exception:
        return False
    if not MLX_MODEL_DIR.exists():
        return False
    _device_info.update(backend="mlx", device="metal", compute_type=None)
    return True


def _try_ctranslate2() -> bool:
    global _ct2_model
    try:
        from faster_whisper import WhisperModel
        import ctranslate2
    except Exception as e:
        print(f"[asr] faster-whisper/ctranslate2 not importable: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return False
    if not CT2_MODEL_DIR.exists():
        print(f"[asr] CT2 model dir not found: {CT2_MODEL_DIR}", file=sys.stderr, flush=True)
        return False
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception as e:
        # This is the single most common way "GPU present, but everything is
        # slow" happens with zero other symptoms: ctranslate2 bundles its own
        # CUDA runtime, separate from PyTorch's, and can fail to find a
        # compatible cuDNN/cuBLAS on Windows (driver/toolkit mismatch, missing
        # MSVC redistributables) without raising past this call -- it just
        # reports 0 devices, and this used to fall back to CPU with nothing in
        # the log to say why. Logging the exception IS the fix for "why is
        # this on CPU".
        print(f"[asr] ctranslate2.get_cuda_device_count() failed: "
              f"{type(e).__name__}: {e} -- falling back to CPU", file=sys.stderr, flush=True)
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
    print(f"[asr] ctranslate2 selecting device={device} compute_type={compute_type} "
          f"(cuda_device_count={cuda_count})", file=sys.stderr, flush=True)
    _ct2_model = WhisperModel(str(CT2_MODEL_DIR), device=device, compute_type=compute_type)
    _device_info.update(backend="ctranslate2", device=device, compute_type=compute_type,
                        cuda_device_count=cuda_count)
    return True


def _select():
    """Pick and load the ASR backend once. Thread-safe: the startup warmup
    thread and a job thread can both reach this, and without the lock both
    would build their own WhisperModel -- two full copies of the weights on
    the GPU, with _ct2_model swapped underneath whichever one is mid-use."""
    global _active
    if _active is not None:
        return
    with _SELECT_LOCK:
        if _active is not None:
            return
        _select_locked()


def _select_locked():
    global _active
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


def unload() -> bool:
    """Free the Whisper weights from the GPU. Returns True if anything was freed.

    The counterpart to chat.unload(). Transcription has always dropped the chat
    model before starting, but nothing ever dropped Whisper before a chat
    request, so the two peaks overlapped in one direction only -- which is what
    puts a T4 over the line once the chat model is unquantized.

    CTranslate2 allocates OUTSIDE PyTorch's caching allocator, so this memory is
    invisible to torch.cuda.memory_allocated() and unreachable by
    torch.cuda.empty_cache(): on the reported OOM, ~4.1 GiB of the card was held
    here while PyTorch could only see (and only report) its own 10.4 GiB.
    Releasing it means dropping the model object itself.

    The next transcribe_chunk() reloads lazily -- a few seconds -- so this only
    trades reload time, never correctness."""
    global _ct2_model, _active
    with _SELECT_LOCK:
        if _ct2_model is None:
            return False
        # Drop the reference and let CTranslate2's destructor release the
        # device memory. Deliberately NOT calling ctranslate2's explicit
        # unload_model() first: that tears the model down and the destructor
        # then runs over an already-torn-down object. A double teardown in a
        # native extension is not something a Python try/except can contain --
        # it takes the worker process with it, which from the browser looks
        # like the request failing and the backend restarting rather than an
        # error being handled. One teardown, via refcount, is enough.
        _ct2_model = None
        _active = None      # force a fresh _select() (and reload) next time
    import gc
    gc.collect()
    print("[mem] unloaded Whisper (ctranslate2) from the GPU", file=sys.stderr, flush=True)
    return True


def backend_info() -> dict:
    """Diagnostic snapshot of which ASR backend/device/compute_type actually
    got selected -- call active_backend() (or transcribe once) first if this
    returns {"loaded": False}; it deliberately does not force a load itself,
    same reasoning as pipeline.model_dir()/model_available()."""
    if _active is None:
        return {"loaded": False}
    return {"loaded": True, **_device_info}


def _temperatures():
    """Whisper's temperature-fallback ladder.

    When a decode trips compression_ratio_threshold (i.e. its output is
    suspiciously repetitive) or the logprob threshold, Whisper RE-DECODES the
    same audio at the next temperature up. Six entries therefore means one
    pathological chunk can be decoded six times over -- and repetitive output
    is exactly what this project's audio produces, so the chunks that
    hallucinate repeats are also the ones burning the most GPU time. The two
    reported symptoms (duplicate text, slow processing) are the same event
    seen from two sides.

    Shortening the ladder (WHISPER_TEMPERATURES="0.0,0.2,0.4") caps that worst
    case at the cost of giving up earlier on genuinely hard audio. Default is
    unchanged from Whisper's own; measure with the [asr] fallback log line
    below before trading accuracy for it."""
    raw = os.environ.get("WHISPER_TEMPERATURES")
    if not raw:
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return [float(x) for x in raw.split(",") if x.strip()]


def _log_fallback(segments):
    """Report chunks that needed a temperature fallback -- the per-chunk cost
    multiplier that is otherwise completely invisible."""
    hits = [s for s in segments if (s.get("temperature") or 0) > 0]
    if hits:
        worst = max(s["temperature"] for s in hits)
        print(f"[asr] temperature fallback fired on {len(hits)}/{len(segments)} "
              f"segment(s) in this chunk (up to T={worst}) -- the chunk was "
              f"decoded more than once", file=sys.stderr, flush=True)


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
            temperature=tuple(_temperatures()), compression_ratio_threshold=2.4,
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
                        "temperature": s.get("temperature"),
                        "words": words or None})
        _log_fallback(out)
        return out

    # ctranslate2 / faster-whisper -- pipeline.py already VAD-chunked the
    # audio, so vad_filter is off here to avoid re-segmenting a chunk that's
    # already speech-only.
    segments, _info = _ct2_model.transcribe(
        audio, language="fa", task="transcribe", beam_size=BEAM_SIZE,
        temperature=_temperatures(), compression_ratio_threshold=2.4,
        no_speech_threshold=0.45, condition_on_previous_text=False,
        vad_filter=False, word_timestamps=word_timestamps)
    out = []
    for s in segments:
        words = [{"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word}
                 for w in (getattr(s, "words", None) or [])] if word_timestamps else None
        out.append({"start": s.start, "end": s.end, "text": s.text,
                    "confidence": _confidence(getattr(s, "avg_logprob", None)),
                    "temperature": getattr(s, "temperature", None),
                    "words": words or None})
    _log_fallback(out)
    return out
