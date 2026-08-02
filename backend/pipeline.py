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
import bisect
import difflib
import os
import threading
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DIARIZER = os.environ.get("DIARIZER", "pyannote")  # "pyannote" | "resemblyzer" | "off"

# Diarization pipeline, best first. community-1 (pyannote.audio 4.x, Sept 2025)
# beats the legacy 3.1 on 10 of 12 published benchmarks -- most relevantly for
# meeting audio recorded over a laptop mic: AMI-SDM 22.7 -> 19.9 DER and
# AliMeeting 24.5 -> 20.3. It ties on VoxConverse and is slightly worse on
# REPERE (7.9 -> 8.9). The win comes from swapping AgglomerativeClustering for
# VBx (Bayesian HMM over x-vectors) + PLDA, which is specifically a
# speaker-CONFUSION fix -- the failure mode this pipeline actually hits.
# Both are gated, and terms must be accepted PER MODEL, so community-1 falling
# through to 3.1 is expected until that is done -- hence a list, not a swap.
# Pin one explicitly with PYANNOTE_PIPELINE.
PYANNOTE_PIPELINES = ["pyannote/speaker-diarization-community-1",
                      "pyannote/speaker-diarization-3.1"]

# community-1 additionally emits an "exclusive" diarization in which at most
# one speaker is active at any instant. Regular diarization marks overlapped
# speech as BOTH speakers at once, so a word landing in an overlap region gets
# its label from whichever turn happens to overlap it more -- effectively a
# coin flip mid-sentence, which is what shreds a turn across two speakers.
# pyannote ships this specifically to reconcile diarization with ASR word
# timestamps, which is exactly what _split_on_speaker does. Off => use the
# regular (overlap-preserving) diarization.
PYANNOTE_EXCLUSIVE = os.environ.get("PYANNOTE_EXCLUSIVE", "1") != "0"

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

# Cut ASR chunks at diarization turn boundaries so each chunk holds exactly one
# speaker (see _speaker_chunks). 0 = chunk by VAD alone and reconcile speakers
# per word afterwards, the older behaviour.
SPEAKER_AWARE_CHUNKS = os.environ.get("ASR_SPEAKER_CHUNKS", "1") != "0"

# Shortest chunk worth sending to Whisper on its own. Below this a "turn" is
# almost always a boundary sliver, and isolated sub-second audio decodes to
# noise or nothing.
MIN_CHUNK_S = float(os.environ.get("ASR_MIN_CHUNK_S", "0.5"))


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


def asr_diagnostic() -> dict:
    """Diagnostic: loads the ASR backend right now (same code path used
    during transcription) and reports which device/compute_type it actually
    landed on. The single most useful check for "GPU memory is used but
    everything is slow" -- ctranslate2's CUDA detection can fail silently on
    Windows and fall back to CPU with no other visible symptom; see
    asr_engine._try_ctranslate2()."""
    import asr_engine
    backend = asr_engine.active_backend()
    return {"backend": backend, **asr_engine.backend_info()}


_ENCODER = None
_NORMALIZER = None
_PYANNOTE = None
_PYANNOTE_ID = None      # which pipeline id actually loaded, for reporting
_PYANNOTE_TRIED = False

# These lazy loaders are now reachable from TWO threads at once: the startup
# warmup thread and a job thread, if a file is uploaded while warmup is still
# running. Without a lock, the second caller sees a half-initialised global --
# for pyannote that meant silently getting None and falling back to a much
# weaker diarizer, with nothing in the log to say it happened. A lock makes a
# concurrent caller WAIT for the in-flight load and then get the real model.
_PYANNOTE_LOCK = threading.Lock()
# Whether pyannote's tensors are currently on the GPU, and which device to send
# them back to. free_diarizer() parks it on CPU between jobs; _diarize_pyannote()
# restores it. Tracked rather than re-derived so a restore never guesses wrong.
_PYANNOTE_ON_GPU = False
_PYANNOTE_DEVICE = None
_VAD_LOCK = threading.Lock()
_VAD_MODEL = None


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


def free_diarizer() -> bool:
    """Move the diarization pipeline to CPU to free its VRAM for the chat model.
    Returns True if anything was moved.

    Moved, NOT dropped. Dropping it would clear the cached pipeline, so the next
    transcription would call Pipeline.from_pretrained() again -- and that goes
    back to Hugging Face to resolve the repo even when the weights are cached.
    On this deployment the Hub has been observed at ~58 kB/s, so that turns
    "upload a file after chatting" into a job that sits at 0% for a long time
    with nothing in the log, which is exactly the reported hang. Keeping the
    object and only relocating its tensors makes the round trip local and
    instant.

    ONLY pyannote. Whisper is deliberately never touched from here: releasing
    the CTranslate2 model -- whether by dropping the object or via its own
    unload_model() -- destabilised this deployment twice, once as a native
    crash with no Python traceback at all. pyannote is pure PyTorch, so moving
    it is thread-safe and has no native teardown.

    _diarize_pyannote() moves it back before it runs."""
    global _PYANNOTE_ON_GPU
    if _PYANNOTE is None or not _PYANNOTE_ON_GPU:
        return False
    with _PYANNOTE_LOCK:
        if _PYANNOTE is None or not _PYANNOTE_ON_GPU:
            return False
        try:
            import torch
            _PYANNOTE.to(torch.device("cpu"))
            _PYANNOTE_ON_GPU = False
        except Exception as e:
            print(f"[mem] could not move the diarizer to CPU: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            return False
    import gc
    gc.collect()
    _free_gpu()
    return True


def gpu_report() -> dict:
    """What is actually on the card right now.

    Reports PyTorch's own numbers alongside the driver-level total via NVML,
    because the gap between them IS the CTranslate2 allocation -- the block
    that never appears in a torch OOM message and is the usual reason the
    arithmetic in one of those messages doesn't add up."""
    out = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        out["cuda"] = True
        free_b, total_b = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        out["total_gib"] = round(total_b / gib, 2)
        out["free_gib"] = round(free_b / gib, 2)
        out["torch_allocated_gib"] = round(torch.cuda.memory_allocated() / gib, 2)
        out["torch_reserved_gib"] = round(torch.cuda.memory_reserved() / gib, 2)
        used = (total_b - free_b) / gib
        out["used_gib"] = round(used, 2)
        # anything in use that PyTorch does not account for is CTranslate2 &c.
        out["non_torch_gib"] = round(max(0.0, used - out["torch_reserved_gib"]), 2)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


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

def _vad_model():
    """Silero VAD, loaded once. Guarded because the warmup thread and a job
    thread can reach this simultaneously."""
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL
    with _VAD_LOCK:
        if _VAD_MODEL is None:
            from silero_vad import load_silero_vad
            _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def _vad_segments(audio):
    from silero_vad import get_speech_timestamps
    import torch
    return get_speech_timestamps(
        torch.from_numpy(audio), _vad_model(), sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=300, min_speech_duration_ms=250, speech_pad_ms=100,
        return_seconds=False)


def _split_long(start, end, tgt):
    """Hard-split a single span into <=tgt pieces.

    A VAD segment can itself be longer than target_s -- fast or overlapping
    conversation with no >=300ms gap for a long stretch produces exactly one
    giant "speech" region (min_silence_duration_ms=300 never fires). Handing
    that whole span to Whisper in one call feeds it more audio than a single
    ~30s decode window, so the backend has to slide its window forward
    internally. With condition_on_previous_text=False (needed to stop
    in-window repetition loops) the model has no memory of what it already
    transcribed, and its own window-advance heuristic -- driven by predicted
    end/timestamp tokens, less reliable on this fine-tuned non-English model
    -- can fail to advance and re-decode audio it already covered: two
    independently-generated, near-identical (not byte-identical) copies of
    the same speech, each with its own slightly different word timings, so
    they can even get diarized to different speakers. Hard-slicing here keeps
    every chunk within one decode window, at the cost of occasionally cutting
    a chunk boundary mid-word -- the same trade-off already made at every
    VAD-segment boundary elsewhere in this function."""
    if end - start <= tgt:
        return [(start, end)]
    pieces, s = [], start
    while s < end:
        e = min(s + tgt, end)
        pieces.append((s, e))
        s = e
    return pieces


def _speaker_chunks(segs, index, target_s=24.0):
    """VAD speech cut at diarization turn boundaries, then merged back within
    a single speaker up to target_s. Returns [(start, end, speaker), ...].

    The pipeline used to chunk purely by VAD and only reconcile speakers
    afterwards, per word. That has two costs, and the second is the one that
    shows up in the transcript:

      1. Whisper is handed audio containing a speaker change and asked to
         decode it as one continuous utterance -- different voice, different
         prosody, often mid-sentence. It is being fed something it was not
         trained to model.
      2. Recovering the boundary afterwards depends on word timestamps landing
         on the right side of it. They frequently do not: a handover can be
         sub-word, and _split_on_speaker then absorbs the short side into its
         neighbour as if it were jitter. On the scored sample, one emitted
         segment covered two reference speakers for exactly this reason.

    Cutting first makes each chunk single-speaker by construction: Whisper
    gets clean audio, and the speaker label is known rather than inferred, so
    no word-level alignment pass is needed for it either.

    The trade is that a diarization error is now baked in -- there is no later
    word-level step that could partially recover from it. That is the right
    trade only because the labels come from community-1's exclusive
    diarization; set ASR_SPEAKER_CHUNKS=0 to go back to VAD-only chunking."""
    if not segs or not index:
        return []
    tgt = int(target_s * SAMPLE_RATE)
    min_len = int(MIN_CHUNK_S * SAMPLE_RATE)

    # every turn edge is a candidate cut point
    bounds = sorted({int(round(v * SAMPLE_RATE))
                     for t in index.turns for v in (t[0], t[1])})
    pieces = []
    for seg in segs:
        s, e = int(seg["start"]), int(seg["end"])
        lo, hi = bisect.bisect_right(bounds, s), bisect.bisect_left(bounds, e)
        cuts = [s] + bounds[lo:hi] + [e]
        for a, b in zip(cuts, cuts[1:]):
            if b > a:
                pieces.append([a, b, index.assign(a / SAMPLE_RATE, b / SAMPLE_RATE)])
    if not pieces:
        return []

    merged = []
    for a, b, spk in pieces:
        if merged:
            prev = merged[-1]
            # A piece too short to transcribe on its own is a turn-boundary
            # sliver, not a turn. Sending Whisper 0.2s of audio yields noise or
            # nothing, so hand it to the neighbour it is already touching.
            if b - a < min_len and prev[2] != spk:
                prev[1] = b
                continue
            if prev[2] == spk and b - prev[0] <= tgt:
                prev[1] = b
                continue
        merged.append([a, b, spk])

    # a single speaker can still hold the floor for longer than one decode
    # window -- same hard split as the VAD-only path
    return [(x, y, spk) for a, b, spk in merged for x, y in _split_long(a, b, tgt)]


def _asr_chunks(segs, target_s=24.0):
    """Merge VAD segments into <=target_s chunks (good context for Whisper),
    first hard-splitting any single segment that already exceeds target_s."""
    if not segs:
        return []
    tgt = int(target_s * SAMPLE_RATE)
    pieces = [p for seg in segs for p in _split_long(seg["start"], seg["end"], tgt)]
    chunks, cs, ce = [], pieces[0][0], pieces[0][1]
    for s, e in pieces[1:]:
        if e - cs <= tgt:
            ce = e
        else:
            chunks.append((cs, ce)); cs, ce = s, e
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
        # Only AgglomerativeClustering (3.1) has min_cluster_size. community-1
        # clusters with VBx, whose knobs are threshold/Fa/Fb -- injecting a key
        # its parameter tree doesn't define makes instantiate() raise, which
        # would silently drop the threshold override sitting next to it too.
        if "min_cluster_size" in clustering:
            clustering["min_cluster_size"] = int(os.environ["PYANNOTE_MIN_CLUSTER_SIZE"])
            changed = True
        else:
            print("[diarize] PYANNOTE_MIN_CLUSTER_SIZE ignored: this pipeline's "
                 f"clustering has no such parameter (has: {sorted(clustering)}). "
                 "It applies to speaker-diarization-3.1's AgglomerativeClustering, "
                 "not community-1's VBx.", file=sys.stderr, flush=True)
    if changed:
        overrides["clustering"] = clustering

    segmentation = dict(params.get("segmentation", {}) or {})
    if os.environ.get("PYANNOTE_MIN_DURATION_OFF"):
        segmentation["min_duration_off"] = float(os.environ["PYANNOTE_MIN_DURATION_OFF"])
        overrides["segmentation"] = segmentation

    return overrides


def _load_pyannote():
    """Load the diarization pipeline, applying the torch / speechbrain compat
    patches. Thread-safe: concurrent callers block until the first finishes and
    then receive the same pipeline.

    _PYANNOTE_TRIED is deliberately set only AFTER the attempt resolves (in the
    finally below), never before it. It used to be set on entry, which made it
    mean two different things -- "already failed, don't retry" and "a load is
    in flight" -- indistinguishable to a second thread. Once the startup warmup
    ran in its own thread, a job starting during warmup hit exactly that: it
    saw TRIED, got None, and silently degraded to the fallback diarizer with
    no log line at all."""
    global _PYANNOTE, _PYANNOTE_ID, _PYANNOTE_TRIED
    if _PYANNOTE is not None or _PYANNOTE_TRIED:
        if _PYANNOTE is None:
            # a previous attempt genuinely failed -- say so rather than
            # returning None mutely, which is what made this invisible
            print("[diarize] pyannote unavailable (an earlier load attempt "
                 "failed); using the fallback diarizer", file=sys.stderr, flush=True)
        return _PYANNOTE
    # Report the wait rather than appearing frozen: this load can take tens of
    # seconds (and downloads the model on a cold cache), so a job that starts
    # during startup warmup blocks here with nothing else to show for it.
    if not _PYANNOTE_LOCK.acquire(blocking=False):
        print("[diarize] a diarization model load is already in flight "
             "(startup warmup?) -- waiting for it…", file=sys.stderr, flush=True)
        t0 = time.time()
        _PYANNOTE_LOCK.acquire()
        print(f"[diarize] waited {time.time() - t0:.1f}s for that load",
              file=sys.stderr, flush=True)
    try:
        # re-check: another thread may have completed the load while we waited
        if _PYANNOTE is not None or _PYANNOTE_TRIED:
            return _PYANNOTE
        return _load_pyannote_locked()
    finally:
        _PYANNOTE_LOCK.release()


def _load_pyannote_locked():
    global _PYANNOTE, _PYANNOTE_ID, _PYANNOTE_TRIED, _PYANNOTE_ON_GPU, _PYANNOTE_DEVICE
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
    import torch
    # pyannote's own checkpoint is trusted, and torch>=2.6 defaults
    # weights_only=True, so it needs full unpickle to load. Patch torch.load
    # only for the duration of THIS load and restore it in the finally below.
    # It used to be replaced permanently and never restored, which (a) forced
    # weights_only=False on every unrelated torch.load in the process for the
    # rest of its life -- the exact footgun torch>=2.6 introduced that default
    # to prevent -- and (b) re-wrapped the already-wrapped function on each
    # subsequent call, nesting a new closure every time.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    try:
        from pyannote.audio import Pipeline
        explicit = os.environ.get("PYANNOTE_PIPELINE")
        candidates = [explicit] if explicit else list(PYANNOTE_PIPELINES)
        pipe, loaded_id, tried = None, None, []
        for model_id in candidates:
            try:
                pipe = Pipeline.from_pretrained(model_id)
            except Exception as e:
                tried.append(f"  {model_id}: {type(e).__name__}: {e}")
                pipe = None
                continue
            if pipe is not None:
                loaded_id = model_id
                break
            # pyannote.audio returns None (no exception) when the model is
            # gated and no token/accepted-terms is available -- this is the
            # single most common reason diarization silently degrades to the
            # weaker resemblyzer fallback, so make it loud.
            tried.append(f"  {model_id}: gated -- no HF token, or its terms "
                        f"not accepted by that token's account "
                        f"(accept at https://hf.co/{model_id})")
        if pipe is None:
            print("[diarize] no pyannote pipeline could be loaded:\n"
                 + "\n".join(tried) +
                 "\nFix: run `huggingface-cli login` with a token from "
                 "https://hf.co/settings/tokens, then accept the terms for the "
                 "model above with that SAME account. Falling back to "
                 "resemblyzer diarization (lower speaker-separation quality).",
                 file=sys.stderr, flush=True)
            return None
        if tried:
            # a better pipeline was available in the list but unusable -- say so,
            # otherwise the quality difference silently looks like a code problem
            print(f"[diarize] preferred pipeline(s) unavailable, fell back to "
                 f"{loaded_id}:\n" + "\n".join(tried), file=sys.stderr, flush=True)

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
        _PYANNOTE_DEVICE = gpu_device
        _PYANNOTE_ON_GPU = gpu_device is not None
        if gpu_device is not None:
            try:
                pipe.to(gpu_device)
            except Exception as e:
                print(f"[diarize] pyannote loaded but failed to move to {gpu_device}: "
                     f"{type(e).__name__}: {e} -- continuing on CPU", file=sys.stderr, flush=True)

        # Peak diarization memory is set by how many sliding windows pyannote
        # batches at once, NOT by the model size -- which is why short files are
        # fine and long ones OOM: a 2h recording produces thousands of windows
        # and the batch size sizes the activation buffers accordingly.
        # Lowering it trades speed for a footprint that stops growing with file
        # length. hasattr-guarded because these attribute names differ between
        # pyannote 3.x and 4.x.
        #
        # 32 is what BOTH speaker-diarization-3.1 and community-1 ship in their
        # own config.yaml. This used to default to 8 -- a 4x cut in windows per
        # forward pass, i.e. 4x the number of GPU round-trips, and diarization
        # runs a 10s window at a 1s step, so the window count is ~10x the audio
        # duration in seconds and that multiplier lands on the whole file. It
        # was set to 8 to stop CUDA OOM back when the ~8GB fp16 chat model
        # stayed resident for the process lifetime; server.py now unloads chat
        # before every transcription, so the constraint that justified it is
        # gone and this is just a self-inflicted slowdown. Drop it back to 8 if
        # a very long file still OOMs on a smaller card.
        batch = int(os.environ.get("PYANNOTE_BATCH", "32"))
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
        print(f"[diarize] {loaded_id} loaded successfully "
             f"(device={gpu_device or 'cpu'})", file=sys.stderr, flush=True)
        _PYANNOTE = pipe
        _PYANNOTE_ID = loaded_id
    except Exception as e:
        print(f"[diarize] pyannote failed to load: {type(e).__name__}: {e} -- "
             "falling back to resemblyzer diarization", file=sys.stderr, flush=True)
        _PYANNOTE = None
    finally:
        # restore the global torch.load, and only now mark the attempt as
        # resolved -- see _load_pyannote()'s docstring
        torch.load = _orig_load
        _PYANNOTE_TRIED = True
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
def _turns_from(ann):
    """(start, end, speaker) triples from either pyannote annotation shape.

    3.x exposes Annotation.itertracks(yield_label=True) -> (segment, track,
    label); 4.x's annotations iterate directly as (turn, speaker) pairs. The
    previous code hardcoded the 4.x pair form, so it would raise on a 3.x
    install -- and pinning back to 3.x is a live option for this project."""
    it = getattr(ann, "itertracks", None)
    if it is not None:
        try:
            return [(seg.start, seg.end, spk) for seg, _, spk in it(yield_label=True)]
        except Exception:
            pass
    return [(turn.start, turn.end, spk) for turn, spk in ann]


def _restore_diarizer():
    """Put pyannote back on the GPU if free_diarizer() parked it on CPU."""
    global _PYANNOTE_ON_GPU
    if _PYANNOTE is None or _PYANNOTE_ON_GPU or _PYANNOTE_DEVICE is None:
        return
    with _PYANNOTE_LOCK:
        if _PYANNOTE is None or _PYANNOTE_ON_GPU or _PYANNOTE_DEVICE is None:
            return
        t0 = time.time()
        _PYANNOTE.to(_PYANNOTE_DEVICE)
        _PYANNOTE_ON_GPU = True
        print(f"[mem] diarizer restored to {_PYANNOTE_DEVICE} in "
              f"{time.time() - t0:.1f}s", file=sys.stderr, flush=True)


def _diarize_pyannote(audio, progress=None):
    import torch
    pipe = _load_pyannote()
    _restore_diarizer()
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

    print(f"[diarize] running {_PYANNOTE_ID or 'pyannote'} on "
         f"{len(audio)/SAMPLE_RATE:.1f}s of audio{' ' + str(kwargs) if kwargs else ''}…",
         file=sys.stderr, flush=True)
    t0 = time.time()
    # Report progress THROUGH diarization. Without this the whole stage is one
    # silent block -- minutes on a long recording -- so "slow" and "wedged" look
    # identical to the user and to the stall watchdog, which then cannot use a
    # timeout short enough to be useful. pyannote calls this hook per internal
    # step with completed/total counts.
    hook = None
    if progress is not None:
        def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
            if total and completed is not None:
                progress(f"Identifying speakers… {step_name} {int(100 * completed / total)}%")
            else:
                progress(f"Identifying speakers… {step_name}")

    payload = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    try:
        dia = pipe(payload, hook=hook, **kwargs) if hook else pipe(payload, **kwargs)
    except TypeError:
        # this pyannote build does not accept hook= -- lose the progress, keep the job
        dia = pipe(payload, **kwargs)
    print(f"[diarize] inference finished in {time.time() - t0:.1f}s",
          file=sys.stderr, flush=True)

    # pyannote 4.x returns a DiarizeOutput carrying one or two annotations;
    # 3.x returns a bare Annotation. Prefer the EXCLUSIVE annotation when the
    # pipeline provides one (community-1 only): in the regular annotation,
    # overlapped speech is emitted as two simultaneous turns, so a word inside
    # an overlap is labelled by whichever of them happens to overlap it more --
    # a near-coin-flip that splits one person's sentence across two speakers.
    # The exclusive annotation resolves overlaps to a single speaker per
    # instant, which is precisely the reconciliation _split_on_speaker needs.
    source, ann = "speaker_diarization", getattr(dia, "speaker_diarization", dia)
    if PYANNOTE_EXCLUSIVE:
        excl = getattr(dia, "exclusive_speaker_diarization", None)
        if excl is not None:
            source, ann = "exclusive_speaker_diarization", excl
    print(f"[diarize] using {source}", file=sys.stderr, flush=True)

    turns = _turns_from(ann)
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


class _TurnIndex:
    """Diarization turns indexed for overlap queries.

    _assign_speaker is called once per WORD and used to scan every turn, so
    cost grew as words x turns -- both proportional to duration, making it
    quadratic in file length. Measured here: ~3.6s of pure Python for a 1h
    recording, ~14.5s for a 2h one. That is not what makes a job slow, but it
    is wasted wall-clock that grows the wrong way, and the fix is small.

    Turns sorted by start, plus a running max-end, bound the scan on both
    sides: nothing starting after the query window can overlap it, and no turn
    whose running max-end is still below the window's start can reach it.
    Because that running max-end is non-decreasing by construction, BOTH
    bounds are binary-searchable, so the kept range is scanned in ascending
    order -- which matters: max() and min() below return the FIRST extreme
    they see, so visiting turns in a different order silently changes the
    winner whenever two speakers overlap a word equally. Ties like that are
    common at a turn boundary, which is exactly where accuracy matters most.
    The original turn order is kept for the same reason.

    Results are identical to the linear scan this replaces -- verified against
    it on randomized overlapping-turn inputs, ties included."""
    __slots__ = ("turns", "_starts", "_maxend", "_sorted", "_pos")

    def __init__(self, turns):
        # keep the caller's order: it is what the linear scan saw, and it
        # decides ties in both the max() and the min() below
        self.turns = list(turns)
        order = sorted(range(len(self.turns)), key=lambda i: self.turns[i][0])
        self._sorted = [self.turns[i] for i in order]
        self._pos = order              # sorted slot -> original index
        self._starts = [t[0] for t in self._sorted]
        self._maxend, m = [], float("-inf")
        for _s, e, _spk in self._sorted:
            m = e if e > m else m
            self._maxend.append(m)

    def __len__(self):
        return len(self.turns)

    def assign(self, seg_start, seg_end):
        if not self.turns:
            return None
        # hi: last turn that starts at or before the window ends.
        # lo: first turn whose running max-end reaches past the window start.
        hi = bisect.bisect_right(self._starts, seg_end) - 1
        lo = bisect.bisect_right(self._maxend, seg_start)
        overlap = defaultdict(float)
        # Accumulate in ORIGINAL turn order, not sorted order, so the dict's
        # insertion order -- and therefore max()'s first-wins tie-break -- is
        # the one the linear scan produced. Candidates per query are few (the
        # turns touching one word), so the sort here is over a handful of items.
        for i in sorted(range(lo, hi + 1), key=self._pos.__getitem__):
            ts, te, spk = self._sorted[i]
            ov = min(seg_end, te) - max(seg_start, ts)
            if ov > 0:
                overlap[spk] += ov
        if not overlap:
            # no overlap (short gap) -> nearest turn by midpoint
            mid = (seg_start + seg_end) / 2
            return min(self.turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2]
        return max(overlap.items(), key=lambda kv: kv[1])[0]


def _assign_speaker(seg_start, seg_end, turns):
    """Speaker whose diarization turns overlap this segment the most."""
    if not turns:
        return None
    if isinstance(turns, _TurnIndex):
        return turns.assign(seg_start, seg_end)
    return _TurnIndex(turns).assign(seg_start, seg_end)


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


# Below this ratio, two adjacent segments are treated as unrelated content --
# calibrated against real transcripts: genuinely different neighbouring
# segments from the same recording scored 0.28-0.42, while a confirmed
# hallucinated repeat (a whole paragraph re-decoded with a handful of words
# swapped) scored 0.74. 0.6 sits well clear of both.
DEDUPE_THRESHOLD = float(os.environ.get("DEDUPE_THRESHOLD", "0.6"))


def _dedupe_repeats(chunk_segments):
    """Drop a segment that is a near-verbatim repeat of the one immediately
    before it, within the SAME transcribe_chunk() call.

    Whisper -- on both the MLX and CTranslate2 backends -- can hallucinate a
    second, slightly reworded copy of a sentence it just decoded: its own
    internal multi-segment timestamp prediction drifts inside one decode and
    it re-emits instead of moving on, rather than the speaker genuinely
    repeating several sentences almost word-for-word. This is a decoder
    artifact, not two audio windows overlapping (VAD-derived chunks never
    share audio samples -- see _asr_chunks), so it always shows up as
    adjacent entries in one chunk's segment list.
    Compare ONLY immediate neighbours -- a real repeated phrase minutes apart
    must survive -- and require near-verbatim similarity (DEDUPE_THRESHOLD),
    not just "same topic", so a person genuinely restating something in
    different words is left alone. Keep the LATER copy: on the samples that
    surfaced this, the earlier copy's word timestamps were the ones that got
    diarized to the wrong speaker."""
    if not chunk_segments:
        return chunk_segments
    out = [chunk_segments[0]]
    for s in chunk_segments[1:]:
        prev_text, text = out[-1]["text"].strip(), s["text"].strip()
        ratio = difflib.SequenceMatcher(None, prev_text, text).ratio() if prev_text and text else 0.0
        if ratio >= DEDUPE_THRESHOLD:
            print(f"[asr] dropped near-duplicate segment (similarity {ratio:.2f}): "
                 f"{prev_text[:60]!r} vs {text[:60]!r}", file=sys.stderr, flush=True)
            out[-1] = s
        else:
            out.append(s)
    return out


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

    # NOTE: chunking now happens AFTER diarization (see below) so the chunk
    # boundaries can be aligned to speaker turns. It used to run here, which
    # forced every speaker boundary to be recovered per-word after the fact.

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
                turns = _diarize_pyannote(audio, progress=progress)
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
        # build the overlap index ONCE per job rather than rescanning the raw
        # turn list for every word (see _TurnIndex)
        if turns:
            turns = _TurnIndex(turns)
        print(f"[diarize] this run used: {diarizer_used} "
             f"({len(turns) if turns else 0} turns)", file=sys.stderr, flush=True)
        if diarizer_used == "none":
            # Diarization was ASKED FOR and produced nothing. The transcript
            # will still be generated, so nothing errors and the UI looks
            # normal -- it just silently labels every segment S1. That is easy
            # to mistake for "the diarizer is bad" rather than "the diarizer
            # never ran", so make the distinction impossible to miss in a log
            # that is otherwise full of third-party warnings.
            print("[diarize] " + "=" * 62 + "\n"
                 "[diarize] NO SPEAKER SEPARATION for this run. Every segment\n"
                 "[diarize] will be labelled S1. This is NOT a diarization\n"
                 "[diarize] quality problem -- no diarizer ran at all.\n"
                 "[diarize] Causes, in order of likelihood:\n"
                 "[diarize]   * pyannote gated: accept the model's terms on HF\n"
                 "[diarize]     with the same account as your cached token\n"
                 "[diarize]   * no fallback installed: pip install resemblyzer\n"
                 "[diarize]   * a load error -- see the [diarize] lines above\n"
                 "[diarize] Check GET /api/diarizer-status for the live answer.\n"
                 "[diarize] " + "=" * 62, file=sys.stderr, flush=True)
        # Diarization is done with the GPU from here on -- the ASR loop below is
        # the only consumer left. Release what it was holding before Whisper
        # starts allocating, otherwise the two peaks overlap for no reason.
        if os.environ.get("PYANNOTE_UNLOAD", "0") == "1":
            _unload_pyannote()
        else:
            _free_gpu()

    # Chunk for ASR. With diarization available, cut at speaker boundaries so
    # each chunk carries exactly one speaker; otherwise fall back to VAD-only
    # chunks with no speaker attached.
    if turns and SPEAKER_AWARE_CHUNKS:
        chunks = _speaker_chunks(vad, turns)
        print(f"[asr] {len(chunks)} speaker-aligned chunks", file=sys.stderr, flush=True)
    else:
        chunks = [(a, b, None) for a, b in _asr_chunks(vad)]
        print(f"[asr] {len(chunks)} VAD chunks (no speaker alignment)",
              file=sys.stderr, flush=True)

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
    for i, (a, b, chunk_spk) in enumerate(chunks):
        _log(f"Transcribing {i+1}/{n}…", progress)
        if b - a < int(0.1 * SAMPLE_RATE):
            continue
        # Word timestamps cost an extra alignment pass, so only pay for them
        # when diarization can actually use them to split a segment -- which a
        # speaker-aligned chunk never needs, since its speaker is already known.
        chunk_segments = asr_engine.transcribe_chunk(
            audio[a:b], SAMPLE_RATE,
            word_timestamps=(bool(turns) and WORD_LEVEL_DIARIZATION
                             and chunk_spk is None))
        chunk_segments = _dedupe_repeats(chunk_segments)
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
            if chunk_spk is not None:
                # speaker-aligned chunk: the label is known by construction,
                # so there is nothing to infer and nothing to split
                _emit(chunk_spk, st, en, raw, None, s.get("confidence"))
                carry_label, carry_count = chunk_spk, len(raw.split())
                continue
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
    out = {
        "duration": round(duration, 2),
        "language": "fa",
        "segments": segments,
        "rawText": raw_text,
        "speakers": speakers,
        "processingTime": round(time.time() - t0, 1),
        "diarizer": diarizer_used,
    }
    # Carry the "no diarizer ran" condition out through the API too, not just
    # the log. A transcript where every line is S1 is indistinguishable from a
    # badly-diarized one unless something says which happened, and nobody
    # reads the backend console before reporting "diarization is broken".
    if diarize and DIARIZER != "off" and diarizer_used == "none":
        out["diarizationWarning"] = (
            "No diarizer ran, so every segment is labelled S1. This is not a "
            "diarization-quality issue. Check GET /api/diarizer-status: most "
            "often pyannote is gated (accept the model terms on Hugging Face "
            "with the same account as your cached token), and the resemblyzer "
            "fallback is not installed.")
    return out
