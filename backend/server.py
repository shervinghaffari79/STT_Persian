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
import threading
import time
import traceback
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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

    def on_segment(seg):
        partial.append(seg)
        # expose a growing snapshot so the client can render the transcript live
        _set(job_id, partial=list(partial),
             speakers=sorted({s["speaker"] for s in partial}, key=lambda x: int(x[1:])))

    correct_fn = correct.correct_segment if gpt_correct else None

    try:
        _set(job_id, state="processing", progress=2, message="Starting…", partial=[])
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


@app.get("/api/health")
def health():
    return {"status": "ok", "model": pipeline.model_dir().name,
            "model_present": pipeline.model_available(), "chat_model": chat.active_model_name(),
            "gpt_correct_available": bool(os.environ.get("OPENAI_API_KEY"))}


@app.get("/api/diarizer-status")
def diarizer_status():
    """Diagnostic: attempts to load pyannote 3.1 right now (same code path
    used during transcription) and reports whether it actually succeeded, so
    you can verify pyannote is really in use -- and see the exact reason if
    it isn't -- without running a full transcription job. Also printed to
    the backend's console/log either way."""
    pipe = pipeline._load_pyannote()
    if pipe is not None:
        return {"pyannote_available": True,
                "detail": "pyannote/speaker-diarization-3.1 loaded successfully; "
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
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
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
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    out = {k: job.get(k) for k in ("state", "progress", "message", "error")}
    # live snapshot for streaming the transcript into the UI
    out["partial"] = job.get("partial", [])
    out["speakers"] = job.get("speakers", [])
    if job.get("state") == "done":
        out["result"] = job.get("result")
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
