#!/usr/bin/env python3
"""
Local Persian ASR pipeline for the web backend.

Best deployable setup from the research/experiments, using the WhisperX-style
"transcribe and diarize independently, then assign speakers by overlap" design
so we get both good WER and good speaker labels:

  ffmpeg 16k mono
    ├─ Silero VAD -> ~24s chunks -> MLX 8-bit Whisper large-v3 (fa) [+ word times]
    └─ pyannote 3.1 speaker diarization (neural segmentation + WeSpeaker embeds)
  -> assign each word/segment the speaker whose turn it overlaps most
  -> Hazm Persian normalization

pyannote 3.1 is gated: it needs a Hugging Face token (cached via
`huggingface_hub login`) with the model's user conditions accepted. If it is
unavailable, the pipeline degrades to a lightweight resemblyzer clustering and,
failing that, a single speaker -- transcription still works either way.
"""
import os
import subprocess
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL_DIR = Path(os.environ.get(
    "MLX_MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "models" / "whisper-large-v3-persian-mlx-q8"),
))
DIARIZER = os.environ.get("DIARIZER", "pyannote")  # "pyannote" | "resemblyzer" | "off"

_ENCODER = None
_NORMALIZER = None
_PYANNOTE = None
_PYANNOTE_TRIED = False


def _log(msg, cb=None):
    if cb:
        cb(msg)


def decode_audio(path: str) -> np.ndarray:
    """Any audio/video container -> 16 kHz mono float32 via ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')[-300:]}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


# ── speech segmentation for transcription ──────────────────────────────────

def _vad_segments(audio):
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    return get_speech_timestamps(
        torch.from_numpy(audio), load_silero_vad(), sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=300, min_speech_duration_ms=250, speech_pad_ms=100,
        return_seconds=False)


def _asr_chunks(segs, target_s=24.0):
    """Merge VAD segments into <=target_s chunks (good context for Whisper)."""
    if not segs:
        return []
    tgt = int(target_s * SAMPLE_RATE)
    chunks, cs, ce = [], segs[0]["start"], segs[0]["end"]
    for seg in segs[1:]:
        if seg["end"] - cs <= tgt:
            ce = seg["end"]
        else:
            chunks.append((cs, ce)); cs, ce = seg["start"], seg["end"]
    chunks.append((cs, ce))
    return chunks


# ── diarization: pyannote 3.1 (preferred) ──────────────────────────────────

def _load_pyannote():
    """Load pyannote 3.1, applying the torch-2.8 / speechbrain-1.1 compat patches."""
    global _PYANNOTE, _PYANNOTE_TRIED
    if _PYANNOTE is not None or _PYANNOTE_TRIED:
        return _PYANNOTE
    _PYANNOTE_TRIED = True
    try:
        # speechbrain 1.1 lazily imports optional integrations (k2/nlp/numba) that
        # aren't buildable on macOS; make those failures non-fatal.
        import speechbrain.utils.importutils as IU
        _orig = IU.LazyModule.ensure_module

        def _safe(self, stacklevel=1):
            try:
                return _orig(self, stacklevel + 1)
            except Exception:
                stub = types.ModuleType(getattr(self, "target", "stub"))
                self.lazy_module = stub
                return stub
        IU.LazyModule.ensure_module = _safe
    except Exception:
        pass
    try:
        import torch
        _orig_load = torch.load  # pyannote ckpt is trusted; allow full unpickle on torch>=2.6
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if pipe is None:
            return None                       # gated / no accepted token
        if torch.backends.mps.is_available():
            try:
                pipe.to(torch.device("mps"))
            except Exception:
                pass
        _PYANNOTE = pipe
    except Exception:
        _PYANNOTE = None
    return _PYANNOTE


def _diarize_pyannote(audio):
    import torch
    pipe = _load_pyannote()
    if pipe is None:
        return None
    dia = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
    turns = [(seg.start, seg.end, spk) for seg, _, spk in dia.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t[0])
    return turns


# ── diarization fallback: resemblyzer ──────────────────────────────────────

def _diarize_resemblyzer(audio, segs):
    global _ENCODER
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    if _ENCODER is None:
        _ENCODER = VoiceEncoder("cpu")
    embs, min_len = [], int(1.6 * SAMPLE_RATE)
    for s in segs:
        a, b = s["start"], s["end"]
        if b - a < min_len:
            c = (a + b) // 2
            a, b = max(0, c - min_len // 2), min(len(audio), c + min_len // 2)
        embs.append(_ENCODER.embed_utterance(preprocess_wav(audio[a:b], source_sr=SAMPLE_RATE)))
    if len(embs) <= 1:
        labels = [0] * len(embs)
    else:
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=0.40).fit_predict(np.vstack(embs))
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE, f"SPK{int(l)}")
            for s, l in zip(segs, labels)]


def _assign_speaker(seg_start, seg_end, turns):
    """Speaker whose diarization turns overlap this segment the most."""
    if not turns:
        return None
    overlap = defaultdict(float)
    for ts, te, spk in turns:
        ov = min(seg_end, te) - max(seg_start, ts)
        if ov > 0:
            overlap[spk] += ov
    if not overlap:
        # no overlap (short gap) -> nearest turn by midpoint
        mid = (seg_start + seg_end) / 2
        return min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2]
    return max(overlap.items(), key=lambda kv: kv[1])[0]


# ── text + output helpers ──────────────────────────────────────────────────

def _normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        from hazm import Normalizer
        _NORMALIZER = Normalizer(correct_spacing=True, remove_diacritics=True,
                                 remove_specials_chars=True, decrease_repeated_chars=True,
                                 persian_style=True, persian_numbers=False,
                                 unicodes_replacement=True, seperate_mi=True)
    return _NORMALIZER


def _words_from(text, start, end, speaker):
    toks = text.split()
    if not toks:
        return []
    step = (end - start) / len(toks)
    return [{"start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2),
             "text": w, "speaker": speaker} for i, w in enumerate(toks)]


def transcribe(path: str, diarize: bool = True, progress=None, on_segment=None) -> dict:
    import mlx_whisper
    t0 = time.time()
    nz = _normalizer()

    _log("Decoding audio…", progress)
    audio = decode_audio(path)
    duration = len(audio) / SAMPLE_RATE

    _log("Detecting speech (VAD)…", progress)
    vad = _vad_segments(audio)
    if not vad:
        vad = [{"start": 0, "end": len(audio)}]
    chunks = _asr_chunks(vad)

    # diarization runs independently of the ASR chunking
    turns = None
    if diarize and DIARIZER != "off":
        _log("Identifying speakers…", progress)
        try:
            if DIARIZER == "pyannote":
                turns = _diarize_pyannote(audio)
            if turns is None:  # pyannote unavailable -> fallback
                turns = _diarize_resemblyzer(audio, vad)
        except Exception:
            turns = None

    # transcribe chunk-by-chunk, assigning the speaker (by overlap) and emitting
    # each finished segment immediately so the UI can stream the transcript.
    label_map, segments = {}, []
    n = len(chunks)
    for i, (a, b) in enumerate(chunks):
        _log(f"Transcribing {i+1}/{n}…", progress)
        if b - a < int(0.1 * SAMPLE_RATE):
            continue
        r = mlx_whisper.transcribe(
            audio[a:b], path_or_hf_repo=str(MODEL_DIR), language="fa", task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
            no_speech_threshold=0.45, condition_on_previous_text=False,
            word_timestamps=False, verbose=None)
        off = a / SAMPLE_RATE
        for s in r.get("segments", []):
            raw = s["text"].strip()
            if not raw:
                continue
            st, en = round(off + s["start"], 2), round(off + s["end"], 2)
            spk_raw = _assign_speaker(st, en, turns) if turns else "SPK0"
            if spk_raw not in label_map:
                label_map[spk_raw] = f"S{len(label_map)+1}"
            spk = label_map[spk_raw]
            text = nz.normalize(raw)
            seg = {"speaker": spk, "start": st, "end": en,
                   "text": text, "words": _words_from(text, st, en, spk)}
            segments.append(seg)
            if on_segment:
                on_segment(seg)

    segments.sort(key=lambda s: s["start"])
    speakers = sorted({s["speaker"] for s in segments}, key=lambda x: int(x[1:]))
    raw_text = "\n\n".join(f"[{s['speaker']}]: {s['text']}" for s in segments)
    return {
        "duration": round(duration, 2),
        "language": "fa",
        "segments": segments,
        "rawText": raw_text,
        "speakers": speakers,
        "processingTime": round(time.time() - t0, 1),
        "diarizer": ("pyannote" if turns and DIARIZER == "pyannote" and _PYANNOTE
                     else ("resemblyzer" if turns else "none")),
    }
