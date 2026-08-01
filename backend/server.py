#!/usr/bin/env python3
"""
FastAPI backend that serves the local Persian SOTA ASR pipeline.

Endpoints:
  POST /api/transcribe   multipart file upload -> {job_id}
  GET  /api/status/{id}  -> {state, progress, message, result?, error?}
  GET  /api/health       -> {status, model}

Transcription runs in a background thread (minutes for long audio); the client
polls /api/status for progress. The ASR/chat models are loaded lazily on the
first request and reused for the process lifetime.

This backend is NOT meant to be exposed directly to the internet. Only the
frontend (Vite, see vite.config.ts / run.sh / run.ps1) binds to 0.0.0.0 and is
network-exposed; it proxies /api/* to this backend over 127.0.0.1, so the
backend only needs to be reachable from the same machine. Host/port here are
still configurable via HOST/PORT env vars (default 127.0.0.1:8000) if you have
a reason to change that, but there is currently no authentication on any
endpoint, so don't bind this to 0.0.0.0 without adding your own access control
in front of it first.
"""
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

import pipeline
import chat
import correct

app = FastAPI(title="Persian SOTA ASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # backend is localhost-only; see module docstring
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job store {job_id: {...}}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _run_job(job_id: str, tmp_path: str, filename: str, diarize: bool, gpt_correct: bool):
    def progress(msg: str):
        # coarse progress: parse "Transcribing i/total" for a percentage
        pct = None
        if msg.startswith("Transcribing "):
            try:
                i, total = msg.split(" ")[1].rstrip("…").split("/")
                pct = 20 + int(78 * int(i) / max(int(total), 1))
            except Exception:
                pass
        elif msg.startswith("Decoding"):
            pct = 5
        elif msg.startswith("Detecting"):
            pct = 10
        elif msg.startswith("Identifying"):
            pct = 15
        _set(job_id, message=msg, **({"progress": pct} if pct is not None else {}))

    partial: list = []
    seen_speakers: set = set()

    def on_segment(seg):
        # The live list is registered in JOBS once (below) and appended to in
        # place. It used to be re-copied on EVERY segment -- list(partial) --
        # alongside a full rescan of every segment's speaker, so publishing n
        # segments cost O(n^2) and grew quadratically with recording length,
        # on the same thread doing the transcription.
        with JOBS_LOCK:
            partial.append(seg)
            if seg["speaker"] not in seen_speakers:
                seen_speakers.add(seg["speaker"])
                JOBS.setdefault(job_id, {})["speakers"] = sorted(
                    seen_speakers, key=lambda x: int(x[1:]))

    correct_fn = correct.correct_segment if gpt_correct else None

    try:
        # register the live list itself; on_segment appends to it in place
        _set(job_id, state="processing", progress=2, message="Starting…",
             partial=partial, speakers=[])
        # The chat LLM caches ~8GB of fp16 weights for the process lifetime once
        # the AI Analysis panel has been used. On a 16GB card that is what turns
        # a long file into a CUDA OOM -- ASR + diarization are left under half
        # the board. Drop it here; the next /api/chat request reloads it lazily.
        if chat.unload():
            print("[mem] unloaded chat model to free GPU for transcription", flush=True)
        result = pipeline.transcribe(tmp_path, diarize=diarize, progress=progress,
                                     on_segment=on_segment, correct_fn=correct_fn)
        result["id"] = job_id
        result["fileName"] = filename
        _set(job_id, state="done", progress=100, message="Complete", result=result)
    except Exception as e:
        traceback.print_exc()
        _set(job_id, state="error", error=str(e), message="Failed")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _warmup():
    """Load the heavy models once at startup instead of inside the first job.

    Everything expensive here is imported lazily inside the pipeline: torch,
    onnxruntime (silero), speechbrain/pyannote, ctranslate2. That deferral is
    what makes the FIRST transcription after a restart behave completely
    differently from every later one -- the backend log shows it plainly, with
    the onnxruntime and speechbrain import warnings only appearing several
    status polls INTO a job rather than at boot.

    On Windows this is worse than the import time alone suggests: loading
    these native extensions pulls in hundreds of DLLs, each one synchronously
    scanned by Defender on first touch, and native module initialisation holds
    the GIL in long stretches that starve uvicorn's event loop. Paying it here,
    on a background thread before any request arrives, turns "the first upload
    randomly hangs, then it's fine forever" into a slow boot -- which is
    honest, and happens while nobody is waiting on a spinner.

    Set WARMUP=0 to skip (e.g. to keep the GPU free until a job actually
    arrives). Failures are logged and ignored: a warmup problem must never
    stop the server from starting, and the real load path will surface it."""
    t0 = time.time()
    try:
        import numpy as np
        print("[warmup] loading VAD + diarization + ASR models…", flush=True)

        # Order matters, and it is deliberately the pipeline's OWN order of use
        # (VAD -> diarize -> ASR), not the reverse. A job submitted while warmup
        # is still running walks that same sequence; loading ASR first -- as
        # this used to -- means warmup is still busy with diarization exactly
        # when a concurrent job reaches its diarization step.
        try:
            # 0.5s of silence is enough to force silero + onnxruntime to load
            pipeline._vad_segments(np.zeros(pipeline.SAMPLE_RATE // 2, dtype=np.float32))
        except Exception as e:
            print(f"[warmup] VAD load failed: {type(e).__name__}: {e}", flush=True)

        if pipeline.DIARIZER == "pyannote":
            try:
                if pipeline._load_pyannote() is None:
                    print("[warmup] WARNING: diarization is NOT available -- "
                          "transcripts will have a single speaker. See the "
                          "[diarize] lines above for the reason.", flush=True)
            except Exception as e:
                print(f"[warmup] pyannote load failed: {type(e).__name__}: {e}", flush=True)

        # resemblyzer is the documented fallback when pyannote is unavailable.
        # If it is missing too there is no fallback at all, and that only
        # surfaces today at the moment a real job needs it -- by which point
        # the transcript is already being produced with one speaker. Probe it
        # at boot so the gap is known before it costs anyone a run.
        try:
            import resemblyzer  # noqa: F401 -- availability probe only
        except Exception:
            print("[warmup] NOTE: resemblyzer is not installed, so there is no "
                  "diarization fallback. If pyannote is unavailable for any "
                  "reason, transcripts get a single speaker instead of degraded "
                  "speaker separation. Fix: pip install resemblyzer", flush=True)

        try:
            print(f"[warmup] ASR backend: {pipeline.asr_diagnostic()}", flush=True)
        except Exception as e:
            print(f"[warmup] ASR load failed: {type(e).__name__}: {e}", flush=True)

        print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[warmup] aborted: {type(e).__name__}: {e}", flush=True)


@app.on_event("startup")
def _on_startup():
    if os.environ.get("WARMUP", "1") == "0":
        print("[warmup] disabled (WARMUP=0) -- models load on first job", flush=True)
        return
    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": pipeline.model_dir().name,
            "model_present": pipeline.model_available(), "chat_model": chat.active_model_name(),
            "gpt_correct_available": bool(os.environ.get("OPENAI_API_KEY"))}


@app.get("/api/asr-status")
def asr_status():
    """Diagnostic: loads the ASR backend right now (same code path used
    during transcription) and reports the actual device/compute_type it
    landed on -- the fastest way to check "GPU memory is used but everything
    is slow" without SSH+profiling. See pipeline.asr_diagnostic()."""
    return pipeline.asr_diagnostic()


@app.get("/api/chat-status")
def chat_status():
    """Diagnostic: loads the chat LLM right now (same code path used by the AI
    Analysis panel) and reports the actual model/device/quantization in use --
    answers "which language model is loaded" and "is it on GPU" without
    SSH+profiling. See chat.backend_info()."""
    chat._ensure()
    return {"backend": chat._active, **chat.backend_info()}


@app.get("/api/diarizer-status")
def diarizer_status():
    """Diagnostic: attempts to load pyannote 3.1 right now (same code path
    used during transcription) and reports whether it actually succeeded, so
    you can verify pyannote is really in use -- and see the exact reason if
    it isn't -- without running a full transcription job. Also printed to
    the backend's console/log either way."""
    pipe = pipeline._load_pyannote()
    if pipe is not None:
        exclusive = hasattr(pipe, "exclusive_speaker_diarization") or \
            pipeline._PYANNOTE_ID.endswith("community-1")
        return {"pyannote_available": True,
                "pipeline": pipeline._PYANNOTE_ID,
                "exclusive_diarization": bool(exclusive and pipeline.PYANNOTE_EXCLUSIVE),
                "detail": f"{pipeline._PYANNOTE_ID} loaded successfully; "
                          "transcriptions will use it for diarization."}
    return {"pyannote_available": False,
            "detail": "pyannote failed to load or is gated (see the backend console for the "
                      "exact reason -- most commonly: no Hugging Face token configured, or "
                      "the model's terms haven't been accepted by that token's account at "
                      "https://hf.co/pyannote/speaker-diarization-3.1). Diarization will fall "
                      "back to the weaker resemblyzer method until this is fixed."}


@app.post("/api/chat")
async def chat_stream(req: Request):
    """Stream a Persian analysis reply from the local chat model."""
    body = await req.json()
    messages = body.get("messages", [])
    transcript = body.get("transcript", "") or ""

    def gen():
        try:
            for tok in chat.stream_chat(messages, transcript):
                yield tok
        except Exception as e:  # surface errors inline so the UI can show them
            yield f"\n[chat error: {e}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/api/chat/title")
async def chat_title(req: Request):
    body = await req.json()
    try:
        return {"title": chat.make_title(body.get("transcript", "") or "")}
    except Exception:
        return {"title": "تحلیل جدید"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), diarize: str = Form("true"),
                     gpt_correct: str = Form("true")):
    if not pipeline.model_available():
        raise HTTPException(500, f"No ASR model found (checked {pipeline.model_dir()})")
    suffix = Path(file.filename or "audio").suffix or ".bin"

    # Stream the upload to disk in chunks, on a worker thread.
    #
    # This was `tmp.write(await file.read())`, which is two separate problems
    # on one line. read() pulls the WHOLE upload into a single bytes object,
    # so a long meeting recording spikes RSS by its entire size; then write()
    # is a synchronous, blocking call sitting inside an `async def`, so it
    # holds the event loop for the duration of that write and uvicorn cannot
    # serve anything else -- not the status polls, not /api/health, nothing.
    # From the browser that is exactly "stuck on Uploading, nothing happens,
    # then suddenly it works": the POST simply has not returned yet, and the
    # frontend leaves its message on 'Uploading to local model…' until it
    # does (see src/services/localAsr.ts). Chunked copy keeps memory flat,
    # and run_in_threadpool keeps the loop free to answer other requests.
    t_up = time.time()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        await file.seek(0)
        await run_in_threadpool(shutil.copyfileobj, file.file, tmp, 1024 * 1024)
    size = Path(tmp_path).stat().st_size
    print(f"[upload] {file.filename!r} {size/1e6:.1f} MB received in "
          f"{time.time() - t_up:.1f}s", flush=True)

    job_id = uuid.uuid4().hex[:12]
    _set(job_id, state="queued", progress=0, message="Queued",
         fileName=file.filename, created=time.time())
    threading.Thread(
        target=_run_job,
        args=(job_id, tmp_path, file.filename or "audio", diarize.lower() != "false",
              gpt_correct.lower() != "false"),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str, since: int = 0):
    """Job status. `since` = number of partial segments the client already
    has; only newer ones are returned, with `partial_total` giving the true
    count so the client can detect a mismatch and resync.

    Without it, every poll re-serialized the ENTIRE growing transcript --
    each segment carrying a per-word timing array -- roughly once a second
    for the whole job. On a long meeting that is megabytes per poll, and it
    gets heavier the longer the job runs. Defaults to 0, which reproduces the
    old full-snapshot behaviour for any client that does not pass it."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        out = {k: job.get(k) for k in ("state", "progress", "message", "error")}
        # copy the slice under the lock: on_segment appends from the worker
        # thread, so handing the live list to the serializer could mutate it
        # mid-encode
        partial = job.get("partial") or []
        since = max(0, min(since, len(partial)))
        out["partial"] = list(partial[since:])
        out["partial_total"] = len(partial)
        out["speakers"] = list(job.get("speakers") or [])
        if job.get("state") == "done":
            out["result"] = job.get("result")
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
