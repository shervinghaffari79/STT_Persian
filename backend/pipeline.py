#!/usr/bin/env python3
"""
Local Persian ASR pipeline for the web backend.

Best deployable setup from the research/experiments, using the WhisperX-style
"transcribe and diarize independently, then assign speakers by overlap" design
so we get both good WER and good speaker labels:

  ffmpeg 16k mono
    ├─ Silero VAD -> ~24s chunks -> Whisper large-v3 (fa) [+ word times]
    └─ pyannote 3.1 speaker diarization (neural segmentation + WeSpeaker embeds)
  -> assign each word/segment the speaker whose turn it overlaps most
  -> Hazm Persian normalization

The actual Whisper inference is delegated to asr_engine.py, which picks
between MLX (Apple-Silicon GPU) and CTranslate2/faster-whisper (CUDA/CPU,
cross-platform) automatically -- see that module for details.

pyannote 3.1 is gated: it needs a Hugging Face token (cached via
`huggingface_hub login`) with the model's user conditions accepted. If it is
unavailable, the pipeline degrades to a lightweight resemblyzer clustering and,
failing that, a single speaker -- transcription still works either way.

An optional `correct_fn(text, speaker, context) -> text` hook (see correct.py)
can be supplied to lightly clean up each segment's text right after it is
produced, before it is streamed to the client or included in the final result.
"""
import os
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DIARIZER = os.environ.get("DIARIZER", "pyannote")  # "pyannote" | "resemblyzer" | "off"

# Word-level speaker assignment needs word timestamps, which cost a second
# alignment pass inside Whisper (cross-attention + DTW, per segment, on top of
# beam search). That is a real and USER-VISIBLE slowdown, not a rounding error.
# DIARIZE_WORD_LEVEL=0 turns it off and falls back to labelling each whole ASR
# segment with its overlap-winner -- much faster, but a segment spanning a
# speaker change then gets one label for both people.
WORD_LEVEL_DIARIZATION = os.environ.get("DIARIZE_WORD_LEVEL", "1") != "0"

# A short run of words bracketed by the SAME speaker on both sides is usually
# diarization jitter rather than a real turn -- but not always: in a meeting a
# 1-3 word interjection ("بله", "درسته") from a second person is a genuine
# turn, and eating it would be wrong. So this is opt-in, 0 = disabled, and the
# absorption below only fires on the same-speaker-both-sides pattern.
SANDWICH_WORDS = int(os.environ.get("DIARIZE_SANDWICH_WORDS", "0"))


def model_dir() -> Path:
    """Best-guess directory of the ASR model that will be used -- mirrors
    asr_engine's auto-detection but WITHOUT forcing a (possibly heavy) model
    load, so a health check stays cheap. See asr_engine.py for the real
    selection used at transcribe time."""
    import asr_engine
    requested = os.environ.get("ASR_BACKEND", "auto").lower()
    if requested in ("mlx", "auto") and asr_engine.MLX_MODEL_DIR.exists():
        try:
            import mlx_whisper  # noqa: F401 -- availability probe only
            return asr_engine.MLX_MODEL_DIR
        except Exception:
            pass
    return asr_engine.CT2_MODEL_DIR


def model_available() -> bool:
    import asr_engine
    return asr_engine.MLX_MODEL_DIR.exists() or asr_engine.CT2_MODEL_DIR.exists()


_ENCODER = None
_NORMALIZER = None
_PYANNOTE = None
_PYANNOTE_TRIED = False


def _log(msg, cb=None):
    if cb:
        cb(msg)


def _free_gpu():
    """Hand cached-but-unused GPU blocks back to the driver.

    PyTorch's caching allocator keeps freed memory reserved for reuse, and
    CTranslate2 (Whisper) allocates OUTSIDE that pool -- so memory pyannote has
    finished with is not visible to Whisper until this is called. Cheap enough
    to run between stages."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unload_pyannote():
    """Drop the cached diarization pipeline and free its memory.

    Diarization finishes entirely before the ASR loop begins, so on a card that
    cannot hold both, keeping pyannote resident buys nothing for the current job
    -- it only saves reload time on the NEXT one. Opt-in via PYANNOTE_UNLOAD=1."""
    global _PYANNOTE, _PYANNOTE_TRIED
    if _PYANNOTE is None:
        return
    _PYANNOTE = None
    _PYANNOTE_TRIED = False  # allow a reload on the next job
    import gc
    gc.collect()
    _free_gpu()
    print("[mem] unloaded pyannote after diarization", file=sys.stderr, flush=True)


def decode_audio(path: str) -> np.ndarray:
    """Any audio/video container -> 16 kHz mono float32 via ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"]
    try:
        # No timeout here previously: a file whose header confuses ffmpeg's
        # probing (seen with some WAV variants -- an unfinalized RIFF size, an
        # unusual bit depth/chunk layout) can make it hang rather than exit
        # non-zero. subprocess.run then blocks forever, the job never reaches
        # state="error", and the only thing the user sees is the "Processing"
        # spinner -- indistinguishable from a slow file except that it never
        # finishes. 10 minutes is generous for decoding alone (no model
        # inference happens here, this is just a format conversion).
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "ffmpeg did not finish decoding this file within 10 minutes. The file's "
            "container is likely malformed (e.g. an incomplete/streamed recording) "
            "rather than genuinely large -- decoding itself is fast relative to "
            "transcription. Try re-exporting or re-recording the file.")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')[-300:]}")
    if len(proc.stdout) == 0:
        # ffmpeg can exit 0 while decoding zero frames -- e.g. a container with a
        # truncated/corrupted sample-index atom (seen with some phone recordings
        # interrupted mid-save). Surface this as a real error instead of silently
        # producing an empty "done" transcription.
        stderr = proc.stderr.decode("utf-8", "ignore")
        hint = ""
        if "truncated" in stderr.lower() or "moov atom not found" in stderr.lower():
            hint = " The file's internal index looks corrupted/incomplete (interrupted recording or transfer)."
        raise RuntimeError(f"No audio could be decoded from this file.{hint} ffmpeg said: "
                          f"{stderr[-300:]}")
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

def _pyannote_overrides(pipe) -> dict:
    """Build an instantiate() override, starting from the model's OWN shipped
    defaults and changing only what an env var explicitly asks for.

    Do not just write out a hand-picked {"clustering": {...}, "segmentation":
    {...}} dict: pyannote's parameter tree can have more keys than the three
    tuned here, and replacing a whole sub-dict risks dropping ones we never
    meant to touch. Fetching pipe.parameters(instantiated=True) first and
    overriding single leaf keys avoids that.

    With no env vars set this returns {} and pipe.instantiate() is never
    called, so the model runs with its own tuning -- the safe default.
    clustering.threshold LOWER -> more distinct speakers (stricter match
    required to merge). clustering.min_cluster_size LOWER -> weaker/shorter
    clusters (e.g. a brief interjection) survive as their own speaker instead
    of being folded away. segmentation.min_duration_off HIGHER -> a longer
    pause is required to end a turn, so brief within-utterance pauses stop
    being read as a speaker change.
    """
    try:
        params = pipe.parameters(instantiated=True)
    except Exception:
        params = {}

    overrides = {}

    clustering = dict(params.get("clustering", {}) or {})
    changed = False
    if os.environ.get("PYANNOTE_THRESHOLD"):
        clustering["threshold"] = float(os.environ["PYANNOTE_THRESHOLD"])
        changed = True
    if os.environ.get("PYANNOTE_MIN_CLUSTER_SIZE"):
        clustering["min_cluster_size"] = int(os.environ["PYANNOTE_MIN_CLUSTER_SIZE"])
        changed = True
    if changed:
        overrides["clustering"] = clustering

    segmentation = dict(params.get("segmentation", {}) or {})
    if os.environ.get("PYANNOTE_MIN_DURATION_OFF"):
        segmentation["min_duration_off"] = float(os.environ["PYANNOTE_MIN_DURATION_OFF"])
        overrides["segmentation"] = segmentation

    return overrides


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
            # pyannote.audio returns None (no exception) when the model is
            # gated and no token/accepted-terms is available -- this is the
            # single most common reason diarization silently degrades to the
            # weaker resemblyzer fallback, so make it loud.
            print(
                "[diarize] pyannote/speaker-diarization-3.1 returned None -- "
                "this means either no Hugging Face token is configured, or "
                "the account behind it hasn't accepted the model's terms. "
                "Fix: run `huggingface-cli login` with a token from "
                "https://hf.co/settings/tokens, then accept the terms at "
                "https://hf.co/pyannote/speaker-diarization-3.1 with that "
                "SAME account. Falling back to resemblyzer diarization "
                "(lower speaker-separation quality) for now.",
                file=sys.stderr, flush=True,
            )
            return None

        overrides = _pyannote_overrides(pipe)
        if overrides:
            try:
                pipe.instantiate(overrides)
                print(f"[diarize] tuning override applied: {overrides}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[diarize] failed to apply tuning override {overrides}: "
                     f"{type(e).__name__}: {e} -- using model defaults", file=sys.stderr, flush=True)

        # PYANNOTE_DEVICE=cpu keeps diarization off the GPU entirely -- slower,
        # but its memory then comes out of system RAM instead of competing with
        # Whisper for the card. The escape hatch for files that OOM regardless
        # of batch size.
        want = os.environ.get("PYANNOTE_DEVICE", "auto").lower()
        if want == "cpu":
            gpu_device = None
        elif torch.cuda.is_available():
            gpu_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            gpu_device = torch.device("mps")
        else:
            gpu_device = None
        if gpu_device is not None:
            try:
                pipe.to(gpu_device)
            except Exception as e:
                print(f"[diarize] pyannote loaded but failed to move to {gpu_device}: "
                     f"{type(e).__name__}: {e} -- continuing on CPU", file=sys.stderr, flush=True)

        # Peak diarization memory is set by how many sliding windows pyannote
        # batches at once, NOT by the model size -- which is why short files are
        # fine and long ones OOM: a 2h recording produces thousands of windows
        # and the default batch (32) sizes the activation buffers accordingly.
        # Lowering it trades a little speed for a footprint that stops growing
        # with file length. hasattr-guarded because these attribute names differ
        # between pyannote 3.x and 4.x.
        batch = int(os.environ.get("PYANNOTE_BATCH", "8"))
        applied = []
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, batch)
                    applied.append(attr)
                except Exception:
                    pass
        if applied:
            print(f"[diarize] batch size {batch} applied to {', '.join(applied)}",
                  file=sys.stderr, flush=True)
        print("[diarize] pyannote/speaker-diarization-3.1 loaded successfully "
             f"(device={gpu_device or 'cpu'})", file=sys.stderr, flush=True)
        _PYANNOTE = pipe
    except Exception as e:
        print(f"[diarize] pyannote failed to load: {type(e).__name__}: {e} -- "
             "falling back to resemblyzer diarization", file=sys.stderr, flush=True)
        _PYANNOTE = None
    return _PYANNOTE


# def _diarize_pyannote(audio):
#     import torch
#     pipe = _load_pyannote()
#     if pipe is None:
#         return None
#     dia = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
#     turns = [(seg.start, seg.end, spk) for seg, _, spk in dia.itertracks(yield_label=True)]
#     turns.sort(key=lambda t: t[0])
#     return turns
def _diarize_pyannote(audio):
    import torch
    pipe = _load_pyannote()
    if pipe is None:
        return None

    # Constraining speaker count is usually a bigger lever than any clustering
    # threshold: it tells the clustering step the answer instead of asking it
    # to guess. PYANNOTE_NUM_SPEAKERS wins outright when the headcount is
    # known; min/max bound it otherwise. All optional -- unset, pyannote
    # clusters freely, same as before.
    kwargs = {}
    if os.environ.get("PYANNOTE_NUM_SPEAKERS"):
        kwargs["num_speakers"] = int(os.environ["PYANNOTE_NUM_SPEAKERS"])
    else:
        if os.environ.get("PYANNOTE_MIN_SPEAKERS"):
            kwargs["min_speakers"] = int(os.environ["PYANNOTE_MIN_SPEAKERS"])
        if os.environ.get("PYANNOTE_MAX_SPEAKERS"):
            kwargs["max_speakers"] = int(os.environ["PYANNOTE_MAX_SPEAKERS"])

    dia = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE},
               **kwargs)
    # pyannote 4.x returns a DiarizeOutput object; the turns live under
    # .speaker_diarization as (turn, speaker) pairs, rather than the old
    # Annotation.itertracks(yield_label=True) API used in pyannote 3.x.
    turns = [(turn.start, turn.end, spk) for turn, spk in dia.speaker_diarization]
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


def _words_even(text, start, end, speaker):
    """Fallback word list when the ASR backend gave us no alignment: spread the
    segment's span evenly across its tokens.

    These timings are INVENTED. They exist so the UI's word highlighting has
    something to work with, and they are close enough for that at normal
    speaking rates -- but they must never drive speaker assignment, because a
    word's apparent position would then be an artifact of token count rather
    than of when it was actually spoken."""
    toks = text.split()
    if not toks:
        return []
    step = (end - start) / len(toks)
    return [{"start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2),
             "text": w, "speaker": speaker, "estimated": True} for i, w in enumerate(toks)]


def _split_on_speaker(words, turns, min_words=2, carry_label=None, carry_count=0):
    """Group consecutive words into runs sharing a speaker.

    This is the point of word timestamps: one ASR segment can span a speaker
    change (someone interjects mid-sentence, or the VAD chunk straddles a
    handover), and labelling the whole segment with a single overlap-winner
    silently attributes one person's words to another.

    Runs shorter than min_words are absorbed into the neighbouring run rather
    than emitted. A single word flipping speaker is nearly always diarization
    jitter at a turn boundary, and honouring it would shred the transcript into
    unreadable one-word rows.

    carry_label/carry_count describe the speaker and word-count of whatever
    was emitted immediately before this call, from a DIFFERENT (earlier)
    Whisper segment. Without them, a short Whisper segment that is entirely
    one speaker internally -- e.g. a one-word aside -- has nothing to compare
    itself against: it IS the only run, so the len(runs) > 1 loop below never
    even runs, and it gets emitted as its own tiny "speaker" no matter how
    obviously it belongs with its neighbours. Carrying the previous call's
    result forward extends the same absorption across that boundary."""
    if not words:
        return []
    labels = [_assign_speaker(w["start"], w["end"], turns) for w in words]

    def group(labels):
        runs = []  # [[speaker, [index, ...]], ...]
        for i, spk in enumerate(labels):
            if runs and runs[-1][0] == spk:
                runs[-1][1].append(i)
            else:
                runs.append([spk, [i]])
        return runs

    # Absorb jitter by RELABELLING the offending words, then regrouping -- not
    # by stitching runs together. Merging run objects directly leaves two
    # adjacent runs carrying the same speaker when a short run in between is
    # absorbed, which then emits as two segments from one person.
    runs = group(labels)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, (spk, members) in enumerate(runs):
            if len(members) >= min_words:
                continue
            prev_len = len(runs[idx - 1][1]) if idx > 0 else (carry_count if carry_label is not None else -1)
            prev_label = runs[idx - 1][0] if idx > 0 else carry_label
            next_len = len(runs[idx + 1][1]) if idx + 1 < len(runs) else -1
            winner = prev_label if (idx == 0 and carry_label is not None and prev_len >= next_len) \
                else (runs[idx - 1][0] if prev_len >= next_len else runs[idx + 1][0])
            if winner is None or winner == spk:
                continue
            for i in members:
                labels[i] = winner
            runs = group(labels)
            changed = True
            break

    # Sandwich absorption (opt-in, DIARIZE_SANDWICH_WORDS): a run bracketed by
    # the SAME speaker on both sides, and short, is usually a wobble in the
    # embedding rather than a real turn -- someone's pitch or loudness shifted
    # mid-sentence. This uses a LARGER word budget than min_words above,
    # because the both-sides-agree pattern is much stronger evidence of jitter
    # than shortness alone. Off by default: a brief interjection from a real
    # second person in a meeting has exactly this shape and must survive.
    if SANDWICH_WORDS > 0:
        changed = True
        while changed and len(runs) >= 3:
            changed = False
            for idx in range(1, len(runs) - 1):
                spk, members = runs[idx]
                prev_spk = runs[idx - 1][0]
                if (prev_spk == runs[idx + 1][0] and prev_spk != spk
                        and len(members) <= SANDWICH_WORDS):
                    for i in members:
                        labels[i] = prev_spk
                    runs = group(labels)
                    changed = True
                    break

    # The case the loops above cannot reach: the WHOLE word list agrees on one
    # speaker (len(runs) == 1), so there is no internal disagreement to
    # trigger absorption, no matter how short it is. This is exactly a short
    # standalone Whisper segment -- absorb it into the carried-in speaker if
    # that speaker had solid evidence and this one does not. The word budget
    # is the sandwich one when enabled, since a standalone short segment
    # between two same-speaker neighbours is the same phenomenon seen across
    # a segment boundary instead of inside one.
    # Budget stays strictly "< min_words" when the sandwich rule is off, so
    # disabling it really is a no-op rather than a quieter behaviour change.
    standalone_ok = (len(runs[0][1]) <= SANDWICH_WORDS if SANDWICH_WORDS > 0
                     else len(runs[0][1]) < min_words)
    if (len(runs) == 1 and carry_label is not None and runs[0][0] != carry_label
            and standalone_ok and carry_count >= min_words):
        runs = [[carry_label, runs[0][1]]]

    return [[spk, [words[i] for i in members]] for spk, members in runs]


def transcribe(path: str, diarize: bool = True, progress=None, on_segment=None,
               correct_fn=None) -> dict:
    import asr_engine
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

    # diarization runs independently of the ASR chunking. `diarizer_used`
    # tracks what ACTUALLY produced `turns` this run (not inferred from
    # global state afterwards), so the result's "diarizer" field is accurate
    # even if pyannote loaded fine earlier but this particular run fell back.
    turns = None
    diarizer_used = "none"
    if diarize and DIARIZER != "off":
        _log("Identifying speakers…", progress)
        if DIARIZER == "pyannote":
            try:
                turns = _diarize_pyannote(audio)
                if turns is not None:
                    diarizer_used = "pyannote"
            except Exception as e:
                print(f"[diarize] pyannote diarization run failed: {type(e).__name__}: {e} -- "
                     "falling back to resemblyzer diarization", file=sys.stderr, flush=True)
                turns = None
        if turns is None and DIARIZER in ("pyannote", "resemblyzer"):
            # either pyannote was never selected, returned nothing (gated/no
            # token), or raised above -- try the lighter local fallback
            try:
                turns = _diarize_resemblyzer(audio, vad)
                if turns is not None:
                    diarizer_used = "resemblyzer"
            except Exception as e:
                print(f"[diarize] resemblyzer fallback also failed: {type(e).__name__}: {e} -- "
                     "continuing with no speaker separation", file=sys.stderr, flush=True)
                turns = None
        print(f"[diarize] this run used: {diarizer_used}", file=sys.stderr, flush=True)
        # Diarization is done with the GPU from here on -- the ASR loop below is
        # the only consumer left. Release what it was holding before Whisper
        # starts allocating, otherwise the two peaks overlap for no reason.
        if os.environ.get("PYANNOTE_UNLOAD", "0") == "1":
            _unload_pyannote()
        else:
            _free_gpu()

    # transcribe chunk-by-chunk, assigning the speaker (by overlap) and emitting
    # each finished segment immediately so the UI can stream the transcript.
    label_map, segments = {}, []
    recent_context = []  # last few corrected "Sx: text" lines, for correct_fn context
    # raw (pre-label-map) speaker id + word count of whatever was emitted last,
    # carried ACROSS Whisper segments and VAD chunks so _split_on_speaker can
    # absorb a short standalone segment into its neighbour even when that
    # neighbour came from a different transcribe_chunk() call entirely
    carry_label, carry_count = None, 0
    n = len(chunks)
    for i, (a, b) in enumerate(chunks):
        _log(f"Transcribing {i+1}/{n}…", progress)
        if b - a < int(0.1 * SAMPLE_RATE):
            continue
        # Word timestamps cost an extra alignment pass, so only pay for them
        # when diarization can actually use them to split a segment.
        chunk_segments = asr_engine.transcribe_chunk(
            audio[a:b], SAMPLE_RATE,
            word_timestamps=bool(turns) and WORD_LEVEL_DIARIZATION)
        off = a / SAMPLE_RATE

        def _emit(spk_raw, st, en, raw, words, confidence):
            """Label, normalize, optionally correct, and publish one segment."""
            if spk_raw not in label_map:
                label_map[spk_raw] = f"S{len(label_map)+1}"
            spk = label_map[spk_raw]
            text = nz.normalize(raw)
            if correct_fn:
                # optional per-segment GPT cleanup, applied before the segment
                # is ever surfaced (streamed or in the final result). Pass a
                # short rolling context of prior corrected segments so the
                # model has enough grounding to resolve ambiguous words
                # instead of defaulting to "[نامفهوم]" for lack of context.
                context = "\n".join(recent_context[-4:])
                text = correct_fn(text, spk, context)
            if words is None:
                # correct_fn may rewrite the text, so estimated timings have to
                # be derived from the FINAL text rather than the raw tokens
                words = _words_even(text, st, en, spk)
            else:
                words = [{**w, "speaker": spk} for w in words]
            seg = {"speaker": spk, "start": st, "end": en, "text": text,
                   "words": words, "confidence": confidence}
            segments.append(seg)
            if text:
                recent_context.append(f"{spk}: {text}")
            if on_segment:
                on_segment(seg)

        for s in chunk_segments:
            raw = s["text"].strip()
            if not raw:
                continue
            st, en = round(off + s["start"], 2), round(off + s["end"], 2)
            if not turns:
                _emit("SPK0", st, en, raw, None, s.get("confidence"))
                continue

            # shift word times onto the full-recording timeline before they are
            # compared against diarization turns, which are absolute
            src_words = [{"start": round(off + w["start"], 2),
                          "end": round(off + w["end"], 2),
                          "text": w["text"].strip()}
                         for w in (s.get("words") or []) if w["text"].strip()]
            if not src_words:
                # backend returned no alignment for this segment: fall back to
                # labelling it whole, which is the old behaviour. No per-word
                # count to carry, so approximate evidence strength from the
                # token count -- consistent with the min_words comparisons
                # everywhere else.
                spk_raw = _assign_speaker(st, en, turns)
                _emit(spk_raw, st, en, raw, None, s.get("confidence"))
                carry_label, carry_count = spk_raw, len(raw.split())
                continue

            runs = _split_on_speaker(src_words, turns,
                                     carry_label=carry_label, carry_count=carry_count)
            for spk_raw, ws in runs:
                r_start = min(w["start"] for w in ws)
                r_end = max(w["end"] for w in ws)
                r_text = " ".join(w["text"] for w in ws).strip()
                if not r_text:
                    continue
                # a split segment inherits the parent's confidence: avg_logprob
                # is computed per ASR segment and cannot be re-derived per run
                _emit(spk_raw, round(r_start, 2), round(r_end, 2), r_text, ws,
                      s.get("confidence"))
                carry_label, carry_count = spk_raw, len(ws)

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
        "diarizer": diarizer_used,
    }
