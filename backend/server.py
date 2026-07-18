#!/usr/bin/env python3
"""
FastAPI backend that serves the local Persian SOTA ASR pipeline.

Endpoints:
  POST /api/transcribe   multipart file upload -> {job_id}
  GET  /api/status/{id}  -> {state, progress, message, result?, error?}
  GET  /api/health       -> {status, model}

Transcription runs in a background thread (minutes for long audio); the client
polls /api/status for progress. The MLX Whisper model is loaded lazily on the
first request and reused for the process lifetime.
"""
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

app = FastAPI(title="Persian SOTA ASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local tool; tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job store {job_id: {...}}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _run_job(job_id: str, tmp_path: str, filename: str, diarize: bool):
    n_hint = {"n": 0}

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

    try:
        _set(job_id, state="processing", progress=2, message="Starting…", partial=[])
        result = pipeline.transcribe(tmp_path, diarize=diarize, progress=progress,
                                     on_segment=on_segment)
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
    return {"status": "ok", "model": pipeline.MODEL_DIR.name,
            "model_present": pipeline.MODEL_DIR.exists(), "chat_model": chat.MODEL}


@app.post("/api/chat")
async def chat_stream(req: Request):
    """Stream a Persian analysis reply from the local Qwen3-4B (MLX)."""
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
async def transcribe(file: UploadFile = File(...), diarize: str = Form("true")):
    if not pipeline.MODEL_DIR.exists():
        raise HTTPException(500, f"Model not found at {pipeline.MODEL_DIR}")
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    job_id = uuid.uuid4().hex[:12]
    _set(job_id, state="queued", progress=0, message="Queued",
         fileName=file.filename, created=time.time())
    threading.Thread(
        target=_run_job,
        args=(job_id, tmp_path, file.filename or "audio", diarize.lower() != "false"),
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
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
