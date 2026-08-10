# 🎤 Persian Speech-to-Text Web Application

A modern, responsive web app for converting Persian speech to text — now powered by a **fully local, on-device SOTA ASR pipeline** (no cloud API).

## 🧠 The pipeline (backend)

Uploaded audio is transcribed entirely on-device by the best-performing setup from the research phase:

```
ffmpeg (16 kHz mono)
   ├─ Silero VAD → ~24s chunks → MLX 8-bit Whisper large-v3 (Persian), GPU (Metal)
   └─ pyannote 3.1 speaker diarization (neural segmentation + WeSpeaker embeddings)
   → assign each segment the speaker it overlaps most (WhisperX-style)
   → Hazm Persian normalization (ZWNJ / spacing / char unification)
```

> **Speaker diarization** uses `pyannote/speaker-diarization-3.1`, which is
> **gated**. Accept its terms at <https://hf.co/pyannote/speaker-diarization-3.1>
> and log in once so the token is cached:
> `python3 -c "from huggingface_hub import login; login('hf_...')"`.
> Without a token the backend automatically falls back to a lighter
> resemblyzer-based diarizer (lower speaker accuracy); transcription is unaffected.

Measured on the two benchmark meeting recordings: **~37–44% WER / ~15–18% CER**
on spontaneous, multi-speaker, code-switched Persian (best deployable local
result; the ROVER ensemble + Persian-fair scoring reaches ~36–40% WER offline).
Runs at roughly real-time on an M2; no data leaves the machine.

### Optional: per-segment GPT cleanup pass

If `OPENAI_API_KEY` is set, each ASR segment is additionally passed through an
"expert Persian transcription editor" GPT pass (`backend/correct.py`) before
it's shown to the user — fixing grammar/punctuation/ASR mistakes, normalizing
terminology (keeping English software terms like API/UI/WebSocket in Latin
script), dropping meaningless filler noise, and marking genuinely
unrecoverable phrases as `[نامفهوم]` rather than guessing. Each call is given
a short rolling context of the last few corrected segments so it has enough
grounding to make confident corrections instead of over-using `[نامفهوم]`. A
safety guard rejects any suspicious output (empty, runaway, leaked
meta-commentary) and falls back to the uncorrected ASR text, so this can only
help, never silently break the transcript. Toggle with `gpt_correct=true|false`
on `/api/transcribe`; it's a no-op if the key isn't set.

## ✨ Features

- 🎯 **Local Persian ASR** — fine-tuned Whisper large-v3, GPU-accelerated via MLX
- 🗣️ **Speaker diarization** — automatic speaker separation and labels
- ✏️ **Speaker manager** — rename anonymous speaker ids (S1, S2, …) to real
  names, and merge ids diarization over-segmented for the same person;
  merges are stored as reversible redirects, never rewriting the original
  segments
- ⏱️ **Timestamped segments** with per-word timings; export to SRT / TXT / JSON
- 🤖 **AI analysis panel** — chat over the transcript with a **local Qwen3 chat model** (MLX `Qwen3-4B-Instruct-2507` on Mac, `Qwen3.5-4B` on Windows/Linux — see [Windows Server deployment](#-windows-server--nvidia-gpu-deployment)), streamed, on-device, no cloud
- 🎨 **Modern, responsive UI** (React + Tailwind + Vite) with a **dark/light theme toggle** (follows OS preference until you pick one explicitly, then remembers it)

## 🏗️ Architecture

```
Browser (React/Vite :5000, network-exposed)
   │  POST /api/transcribe  (multipart upload)
   │  GET  /api/status/{id} (poll progress)   ── Vite proxy ──▶  FastAPI 127.0.0.1:8000
   │                                                              (localhost-only)
   ▼                                                              backend/pipeline.py
Transcript + speakers rendered in the middle panel
```
Only the frontend's port needs to be reachable from outside the machine — the
backend is proxied internally and never exposed directly.

## 🚀 Getting Started

### Prerequisites
- **Node.js 16+** and npm
- **Python 3.9+** and **ffmpeg** (`brew install ffmpeg` / `choco install ffmpeg` / `apt install ffmpeg`)
- **macOS + Apple Silicon** — uses MLX on the Metal GPU; needs the model at
  `models/whisper-large-v3-persian-mlx-q8` (inside the repo root).
- **Windows / Linux / other Mac** — uses faster-whisper (CTranslate2) +
  `transformers` instead; see [Windows Server / NVIDIA GPU](#-windows-server--nvidia-gpu-deployment) below.
  Backend selection is automatic (see `backend/asr_engine.py`, `backend/chat.py`).

### Run everything (backend + frontend)

```bash
# one-time: install deps
pip3 install -r backend/requirements.txt
npm install

# start backend (127.0.0.1:8000, internal only) AND frontend (0.0.0.0:5000, exposed) together
./run.sh
```

Then open **http://localhost:5000** (or `http://<this-machine-ip>:5000` from
another device on the network), drop an audio/`.mp4` file, and click
**Transcribe Audio**.

### Or run the two services separately

```bash
# terminal 1 — backend
cd backend && python3 server.py         # FastAPI on http://127.0.0.1:8000 (not exposed)

# terminal 2 — frontend
npm run dev                              # Vite on http://0.0.0.0:5000 (exposed)
```

The frontend proxies `/api/*` to the backend (see `vite.config.ts`).

### Backend API
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/transcribe` | multipart `file` (+ `diarize=true\|false`, `gpt_correct=true\|false`) → `{ job_id }` |
| `GET` | `/api/status/{job_id}` | `{ state, progress, message, result? }` |
| `POST` | `/api/chat` | `{ messages, transcript }` → streamed Persian reply (local Qwen3 chat model) |
| `POST` | `/api/chat/title` | `{ transcript }` → `{ title }` |
| `GET` | `/api/health` | model presence check (`gpt_correct_available` reflects whether `OPENAI_API_KEY` is set) |
| `GET` | `/api/status/{job_id}` (`?since=N`) | pass `since` = segments already held; only newer ones come back, with `partial_total`. Omit for the full snapshot |
| `GET` | `/api/gpu-status` | live VRAM split — `non_torch_gib` is the CTranslate2 block PyTorch can't see — plus `active_jobs` and `chat_loaded` |
| `GET` | `/api/asr-status` | loads the ASR backend now, reports actual device/compute_type (e.g. confirms CUDA isn't silently falling back to CPU) |
| `GET` | `/api/chat-status` | loads the chat LLM now, reports actual model/device/quantization -- "which language model is loaded" answered directly |
| `GET` | `/api/diarizer-status` | loads pyannote now, reports which pipeline (community-1 / 3.1) and whether exclusive diarization is active |

The ASR (Whisper) and chat (Qwen3) models run entirely locally — on MLX/Metal
on a Mac, or CTranslate2/`transformers` on CUDA/CPU elsewhere — nothing is sent
to any external API for those. The one optional exception is the per-segment
GPT cleanup pass above, which sends only that segment's already-transcribed
text (never raw audio) to OpenAI, and only when `OPENAI_API_KEY` is configured.

---

## 🖥️ Windows Server / NVIDIA GPU deployment

The backend auto-detects the platform and swaps in a cross-platform runtime —
same API, same pipeline, no code changes needed:

| | macOS (Apple Silicon) | Windows / Linux |
|---|---|---|
| ASR | MLX Whisper (Metal GPU) | faster-whisper/CTranslate2 (CUDA if present, else CPU int8) |
| Chat | MLX (`mlx-lm`), `Qwen3-4B-Instruct-2507` | `transformers`, `Qwen/Qwen3.5-4B` (CUDA if present, else CPU) |

Windows/Linux intentionally run a **different, newer** chat model
(`Qwen/Qwen3.5-4B`) than the Mac MLX path — this is deliberate, not a
mismatch to fix. Two things about it are easy to get wrong:

- **It needs `HF_CHAT_MODEL`'s `transformers` support.** `Qwen3.5-4B`'s
  `model_type` (`qwen3_5`) may not exist yet in a stable `transformers`
  release — its own model card says to install from `main`:
  `pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"`
  (also needs `torchvision` + `pillow`, even for text-only use — already in
  `requirements.txt`). If the installed version doesn't recognize it,
  `backend/chat.py` raises a clear error naming this exact fix rather than a
  bare `Unrecognized configuration class` traceback.
- **It's a vision-language checkpoint, but that's not a problem in itself.**
  Its `config.json` lists `architectures: ["Qwen3_5ForConditionalGeneration"]`
  (the VLM class) plus a full vision tower — but that field is metadata, not
  what actually loads: `transformers`' `AutoModelForCausalLM` resolves
  `qwen3_5` to `Qwen3_5ForCausalLM`, a dedicated text-only class that never
  instantiates the vision tower (`chat.py` already uses `AutoModelForCausalLM`,
  correctly). What genuinely costs time is that **Qwen3.5 thinks by default**
  — it emits a `<think>...</think>` block (recommended up to 32k–80k tokens
  for hard tasks per its model card) before every reply unless
  `enable_thinking=False` is honored by the chat template. `chat.py` passes
  that already; if it ever silently stops working, every reply pays for a
  full hidden reasoning pass with nothing visibly wrong client-side — exactly
  "got slower, no error." `chat.py` now decodes the raw output once per
  response to detect this and logs `[chat] enable_thinking=False did NOT
  suppress a <think> block ...` if it happens.

Check what's actually loaded — model, device, quantization — with
`GET /api/chat-status`, and watch the backend log for that `<think>`-leak
warning after a real chat message.

### 1. Get the models onto the server
- ASR: `models/whisper-large-v3-persian-ct2-int8/` (CTranslate2 int8 Whisper large-v3
  fine-tuned for Persian) — copy this directory from the repo root, or re-download
  via `huggingface-cli download` if you have the original model id.
- Chat: `Qwen/Qwen3.5-4B` is fetched automatically from Hugging Face the first
  time `backend/chat.py` runs (no manual step, just needs the HF cache to have
  internet access once, and a `transformers` install that recognizes it — see
  above).

### 2. Install dependencies
```powershell
# Install a CUDA build of torch FIRST (adjust cu121 to match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r backend/requirements.txt
npm install
```

### 3. Run
```powershell
.\run.ps1
```
or, from `cmd.exe`:
```bat
run.bat
```
This starts the backend (FastAPI, `127.0.0.1:8000`, **not** network-exposed)
and the frontend (Vite, `0.0.0.0:5000`, network/internet-exposed) the same way
`run.sh` does on macOS/Linux. **Only port 5000 needs to be reachable from
outside the machine** — the frontend proxies API calls to the backend
internally, so the backend never needs to be opened up.

Or run them in two terminals:
```powershell
cd backend; python server.py
# separate terminal (from the repo root):
npm run dev
```

### 4. Open the firewall (only port 5000)
```powershell
New-NetFirewallRule -DisplayName "Persian ASR (5000)" -Direction Inbound `
  -LocalPort 5000 -Protocol TCP -Action Allow
```
Then browse to `http://<server-ip>:5000` from another machine. Port 8000
(the backend) should stay closed/unreachable from outside — it has no
authentication, so it's the frontend's proxy, not a firewall rule, that keeps
it from being hit directly.

> ⚠️ **No authentication exists on the API today.** Anyone who can reach
> port 5000 can submit transcription jobs (GPU time) and, if `OPENAI_API_KEY`
> is set, trigger OpenAI-billed chat/correction calls. If this needs to be
> reachable beyond a trusted network, put a reverse proxy with auth (or a
> VPN/IP allowlist at the firewall/security-group level) in front of it.

### Notes
- **Custom ports:** `$env:BACKEND_PORT` (default 8000) and
  `$env:FRONTEND_PORT` (default 5000), set before running `.\run.ps1`
  (`set BACKEND_PORT=...` / `set FRONTEND_PORT=...` before `run.bat` in
  `cmd.exe`).
- **Force a specific backend** if auto-detection ever guesses wrong:
  `$env:ASR_BACKEND="ctranslate2"`, `$env:CHAT_BACKEND="transformers"` (values:
  `mlx` | `ctranslate2` for ASR, `mlx` | `transformers` for chat, or `auto`).
- **Tesla T4 (or other Turing-class GPU):** the ASR backend automatically uses
  CTranslate2's `int8_float16` compute type on CUDA, which runs the
  already-int8-quantized model directly on the T4's INT8 Tensor Cores instead
  of dequantizing to float16 first -- faster than plain `float16` on this GPU
  class. Override with `$env:CT2_COMPUTE_TYPE="float16"` if you ever need to
  compare. 16GB VRAM comfortably fits both models loaded together (ASR ~2GB,
  chat ~8GB in fp16), so both can stay resident without swapping.
- **No GPU?** Both cross-platform backends fall back to CPU automatically —
  slower, but fully functional (this was benchmarked at roughly real-time to
  2x real-time for ASR on CPU earlier in this project).
- **pyannote diarization** needs a Hugging Face token with the model's terms
  accepted (`huggingface-cli login`, then accept the terms). The backend tries
  `speaker-diarization-community-1` first and falls back to `speaker-diarization-3.1`,
  then to a lighter local clustering method. **Terms are accepted per model**, so
  accept both:
  <https://hf.co/pyannote/speaker-diarization-community-1> and
  <https://hf.co/pyannote/speaker-diarization-3.1>.
  Check which one is actually live with `GET /api/diarizer-status`.

### Diarization tuning

`community-1` (pyannote.audio 4.x) is preferred because it beats 3.1 on 10 of
12 published benchmarks — for meeting audio the relevant ones are AMI-SDM
(22.7 → 19.9 DER) and AliMeeting (24.5 → 20.3). It replaces 3.1's
agglomerative clustering with VBx + PLDA, which specifically reduces *speaker
confusion*. It also emits an **exclusive** diarization (one speaker at any
instant) that the pipeline uses by default to align words to speakers —
without it, a word inside overlapped speech is labelled by whichever
overlapping turn covers it more, which is what splits one person's sentence
across two speaker labels.

| Env var | Default | Effect |
|---|---|---|
| `PYANNOTE_PIPELINE` | *(auto)* | Pin one pipeline instead of trying community-1 → 3.1 |
| `PYANNOTE_EXCLUSIVE` | `1` | Use exclusive diarization when available (`0` = allow overlaps) |
| `PYANNOTE_BATCH` | `32` | Windows per forward pass. Upstream default; lower only if a long file OOMs |
| `PYANNOTE_NUM_SPEAKERS` | *(unset)* | Exact headcount — the single biggest accuracy lever when known |
| `PYANNOTE_MIN_SPEAKERS` / `PYANNOTE_MAX_SPEAKERS` | *(unset)* | Bound the headcount when the exact number isn't known |
| `PYANNOTE_THRESHOLD` | *(model default)* | Clustering threshold; **lower ⇒ more distinct speakers** |
| `PYANNOTE_MIN_CLUSTER_SIZE` | *(model default)* | 3.1 only (community-1's VBx has no such knob; it is ignored with a log line) |
| `PYANNOTE_UNLOAD` | `0` | Free pyannote's VRAM before the ASR loop starts |
| `PYANNOTE_DEVICE` | `auto` | `cpu` keeps diarization off the GPU entirely |

If speakers are still being merged, set `PYANNOTE_NUM_SPEAKERS` when you know
the headcount — it tells clustering the answer instead of asking it to infer
one, and outperforms any threshold tuning.

### Startup warmup

Every heavy dependency (torch, onnxruntime/silero, speechbrain/pyannote,
ctranslate2) is imported lazily inside the pipeline, so without warmup the
**first** transcription after a restart pays all of it while the user watches
a spinner — on Windows that also means hundreds of DLLs being scanned by
Defender on first touch, and native module init holding the GIL in long
stretches that starve uvicorn's event loop. The backend now loads them on a
background thread at startup (`[warmup] …` lines in the log). Set `WARMUP=0`
to skip it and keep the GPU free until a job actually arrives.

> If uploads appear to hang and then "suddenly work", check the `[upload]`
> log line for the size and duration. The upload is now streamed to disk in
> 1 MB chunks on a worker thread; previously the whole file was read into RAM
> and written synchronously inside the async handler, which stalled the event
> loop for the entire write — measured at **903 ms of total starvation per
> 0.9 s of write**, scaling with file size, during which *no* request was
> served, including the status polls the UI depends on.

### Measuring quality (don't eyeball it)

```bash
python backend/evaluate.py ground_truth.txt exported_transcript.txt
```

Reports WER/CER after Persian-aware normalization (yeh/kaf folding, ZWNJ,
diacritics, punctuation) plus an insertion/deletion/substitution split, and
compares speaker counts. The reference format is alternating `Speaker N` lines
and their text; the prediction is this backend's `[S1]: …` export. A partial
export is scored against the best-matching prefix of the reference, and the
covered fraction is printed — check it before trusting the number.

**Read the I/D/S split, not just the WER.** A duplicated segment and a genuine
accuracy regression both raise WER, but they have opposite fixes: duplication
shows up as insertions with substitutions flat. On the first scored sample,
one duplicated segment accounted for **54.3% → 34.3% WER** and **33.8% → 14.7%
CER** on its own.

### Speaker-aligned ASR chunking

With diarization available, ASR chunks are cut at speaker-turn boundaries so
each chunk contains exactly one speaker (`_speaker_chunks`). Previously chunks
came from VAD alone and speakers were reconciled per word afterwards, which
meant Whisper decoded across speaker changes as if they were one utterance and
the boundary had to be recovered from word timestamps — on the scored sample a
single emitted segment covered two reference speakers that way.

The trade-off: a diarization error is now baked in, with no later word-level
step that could partly recover from it. Set `ASR_SPEAKER_CHUNKS=0` to return to
VAD-only chunking. Sub-`ASR_MIN_CHUNK_S` slivers at turn boundaries are folded
into their neighbour rather than sent to Whisper alone.

### ASR speed/accuracy knobs

Decode cost is dominated by beam width and by the temperature-fallback ladder:
when a decode trips `compression_ratio_threshold` (repetitive output) Whisper
**re-decodes the same audio** at the next temperature, up to 6 times. The
backend now logs `[asr] temperature fallback fired …` whenever that happens —
check for it before trading accuracy for speed.

| Env var | Default | Effect |
|---|---|---|
| `WHISPER_BEAM_SIZE` | `5` | `1` (greedy) is the largest single ASR speedup, at some WER cost |
| `WHISPER_TEMPERATURES` | `0.0,0.2,0.4,0.6,0.8,1.0` | Shorten to cap worst-case re-decoding |
| `DIARIZE_WORD_LEVEL` | `1` | `0` skips Whisper's word-alignment pass (~10–20% faster, coarser speaker splits). Unused when speaker-aligned chunking is on — the label is already known |
| `ASR_SPEAKER_CHUNKS` | `1` | `0` reverts to VAD-only chunks + per-word speaker reconciliation |
| `ASR_MIN_CHUNK_S` | `0.5` | Shortest standalone chunk; shorter turn slivers fold into a neighbour |
| `CONSOLIDATE_SEGMENTS` | `1` (on) | Merge consecutive same-speaker segments into one turn, so a label changes only when someone else speaks |
| `DEDUPE_THRESHOLD` | `0.6` | Similarity above which an adjacent repeated segment is dropped |
| `CT2_COMPUTE_TYPE` | `int8_float16` on CUDA | Override the compute type |

### GPU memory: who holds what

**Everything stays resident on the GPU.** Nothing is unloaded or moved for
chat — the room comes from running the chat model in **4-bit (NF4)** instead:

```
fp16   Whisper 4.09 + pyannote 1.10 + chat 7.97 = 13.16 / 14.83 GB -> 1.67 GB free  OOMs
4-bit  Whisper 4.09 + pyannote 1.10 + chat ~2.8 =  ~8.0 / 14.83 GB -> ~6.8 GB free  OK
```

**4-bit is not the same trade as 8-bit.** `LLM.int8()` adds a mixed-precision
decomposition for outlier channels and is genuinely *slower* than fp16 at this
model size, which is why 8-bit was never made the default. NF4 dequantizes to
the compute dtype with no outlier path, and single-stream decoding on a T4 is
bandwidth-bound rather than compute-bound — a quarter of the weight traffic
usually more than pays for the dequantization.

The honest cost is **accuracy**, not speed: 4-bit is the lossiest of the three,
and this is a Persian summarization/QA task where that can show up as weaker
recall of specific details. If answers degrade noticeably, `CHAT_4BIT=0`
`CHAT_8BIT=1` is the middle option.

Compute dtype is fp16, not bf16: the T4 is Turing (sm_75) with no bf16
hardware. bitsandbytes 4-bit needs sm_75+, which the T4 satisfies exactly.

> `bitsandbytes` must be installed or the quantized load falls back to fp16 —
> `chat.py` logs it, and the symptom is an OOM later rather than an install
> error. It is in `requirements.txt`.

**Nothing is unloaded at all any more.** All three models stay resident for the
process lifetime, including through a transcription:

```
Whisper 4.09 + pyannote 1.10 + chat 2.80 (4-bit) = 7.99 / 14.83 GB -> 6.84 GB free
```

The chat model used to be dropped before every job, which was correct at fp16
(the same three came to 13.16 GB, leaving 1.67 GB for diarization activations)
but at 4-bit only costs a ~10s reload on the first chat message afterwards.
`UNLOAD_CHAT_FOR_TRANSCRIBE=1` restores it — worth doing if you go back to
fp16/8-bit weights, or if a very long recording pushes diarization (batch 32,
activations scale with duration) into the remaining headroom.

> ⚠️ **Recovering from an OOM must happen outside the `except` block.** While a
> handler runs, Python holds the exception as the *current* exception; its
> traceback keeps the generating frame alive, and that frame holds the model. An
> `unload()` called inside the handler clears the module global and frees
> nothing. Clearing `e.__traceback__` is *not* sufficient; only leaving the
> handler releases it.

**Nothing gets stuck.** A watchdog fails any job that stops progressing for
`JOB_STALL_TIMEOUT` seconds (default 300). Both long stages — ffmpeg decoding
and diarization — report progress as they run, so the longest legitimate
silence is a model load.

`GET /api/gpu-status` shows the live split; `non_torch_gib` is the CTranslate2
block PyTorch cannot see. `GET /api/chat-status` reports
`"quantization": "4bit"` when this is working.

**VRAM creep across chats.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is now set by `server.py` before anything imports torch (PyTorch reads it once,
when it initialises its CUDA allocator, and ignores later changes — so an
explicit value in your environment still wins, but you no longer have to set
one).

Without it the caching allocator keeps fixed-size blocks, and a request whose
tensors are a different *shape* cannot reuse them — it takes fresh segments and
reserved memory ratchets up while allocated memory stays flat. That is why more
questions in the **same** chat cost nothing (identical prompt shape every time,
since history is not resent) while opening a **new** chat raises VRAM: a new
chat first calls `/api/chat/title`, which generates with a very different shape
(~500-character prompt, 24 new tokens) and strands blocks the following
full-transcript request cannot use. Each generation now also returns its
finished KV cache and activations to the driver.

### Chat model knobs

The chat LLM runs **unquantized** (fp16 on CUDA). 8-bit was previously the
default, which traded generation speed for VRAM without saying so: bitsandbytes
`LLM.int8()` is a *footprint* optimization, not a speed one — it dequantizes on
the fly plus runs a mixed-precision path for outlier channels, and only pays
for itself above a hidden-dimension crossover. Qwen3.5-4B (hidden size 2560)
sits below it, so 8-bit was actively slower than fp16 there; a widely-cited
7B/A100 comparison shows ~6.7 tok/s at 8-bit vs ~16.7 tok/s at fp16.

fp16 rather than bf16 on CUDA is deliberate: the checkpoint is bf16, but bf16
needs Ampere (sm_80+) and the T4 is Turing (sm_75) — no bf16 hardware, full
fp16 tensor-core support.

| Env var | Default | Effect |
|---|---|---|
| `CHAT_4BIT` | `1` (on) | NF4 4-bit, ~8 GB → ~2.8 GB, so Whisper + diarizer + chat all stay on the GPU. `0` falls through to `CHAT_8BIT`/fp16 |
| `CHAT_8BIT` | `0` (off) | 8-bit — only used when `CHAT_4BIT=0`. Slower than fp16 at this model size |
| `CHAT_DEVICE` | `auto` | `cpu` keeps the chat model off the GPU entirely; `cuda` forces it on |
| `UNLOAD_CHAT_FOR_TRANSCRIBE` | `0` (off) | `1` drops the chat model before each transcription (only needed at fp16/8-bit) |
| `JOB_STALL_TIMEOUT` | `300` | Seconds without progress before a job is failed so the UI stops waiting |
| `DECODE_TIMEOUT` | `600` | Seconds ffmpeg may take before the decode is abandoned as a malformed container |
| `ASR_SLOW_CHUNK_WARN` | `20` | Seconds after which a single ASR chunk is logged as slow, with its audio timestamps |
| `CHAT_STATELESS` | `1` (on) | Each question answered independently — prior turns are not sent. `0` keeps the full conversation |
| `ASR_DEVICE` | `auto` | `cpu` keeps Whisper off the GPU (frees ~4.1 GB for chat, much slower transcription) |
| `CHAT_MAX_TOKENS` | `350` | Hard ceiling on a reply. Backstop for the brevity instruction; raise it if answers are cut mid-sentence |
| `CHAT_BACKEND` | `auto` | `mlx` \| `transformers` if auto-detection guesses wrong |
| `HF_CHAT_MODEL` | `Qwen/Qwen3.5-4B` | Override the Windows/Linux chat model |

The chat model is unloaded before every transcription (see `server.py`), so at
8-bit it only contends with ASR/diarization if a chat request arrives *mid-job*
— and at ~5 GB there is room for that. If it still OOMs, `CHAT_DEVICE=cpu` keeps
it off the GPU entirely. Confirm what's actually loaded with
`GET /api/chat-status` — it reports `"quantization": "8bit"` when this is
working, and `null` if it silently fell back to fp16.

> **torchcodec on Windows:** pyannote.audio 4.x dropped the `soundfile`/`sox`
> audio backends, so it decodes via torchcodec. If the log shows
> `Could not load libtorchcodec`, torchcodec doesn't match your torch build —
> install the matching version ([compatibility table](https://github.com/pytorch/torchcodec#installing-torchcodec)).
> This pipeline feeds pyannote an in-memory waveform, so diarization still
> works, but any file-path code path in pyannote will not.
- **`gpt_correct`** (per-segment GPT cleanup) is fully cross-platform already —
  it just calls the OpenAI API — set `OPENAI_API_KEY` as an environment variable.

---

### Frontend details

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:shervinghaffari79/STT_Persian.git
   cd STT_Persian
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5000` (bound to
   `0.0.0.0`, so also reachable from other devices on the network)

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 🛠️ Tech Stack

- **Frontend Framework**: React 19.2.3
- **Build Tool**: Vite 7.2.4
- **Styling**: Tailwind CSS 4.1.17
- **Audio Visualization**: WaveSurfer.js 7.12.6
- **Language**: TypeScript 5.9.3
- **Icons**: Lucide React 1.8.0

## 📁 Project Structure

```
src/
├── App.tsx              # Main application component
├── components/
│   ├── AudioPanel.tsx        # Waveform playback + seek
│   ├── TranscriptPanel.tsx   # Segment list, click-to-seek, per-word timing
│   ├── SpeakerManager.tsx    # Rename/merge diarized speaker ids
│   ├── ChatPanel.tsx         # AI analysis chat over the transcript
│   └── Markdown.tsx          # Renders chat replies
├── hooks/
│   └── useTheme.ts      # Dark/light theme state (localStorage + OS preference)
├── services/            # Calls to the FastAPI backend (/api/*)
├── utils/                # clipboard, classnames helpers
└── index.css             # Tailwind + theme tokens

backend/
├── server.py         # FastAPI app, /api/* routes
├── pipeline.py        # VAD chunking → ASR → diarization → speaker assignment
├── asr_engine.py      # Whisper backend selection (MLX / CTranslate2)
├── chat.py            # Qwen3 chat backend selection (MLX / transformers)
└── correct.py         # optional per-segment GPT cleanup pass
```

## 🔧 Configuration

- **Vite Config**: `vite.config.ts` - Build and dev server configuration
- **TypeScript Config**: `tsconfig.json` - TypeScript compiler options
- **Tailwind Config**: Configured via `@tailwindcss/vite` plugin

## 📝 Development

### Code Style
- Uses TypeScript for type safety
- Follows React best practices
- Tailwind CSS for styling

### Running Tests
Tests configuration can be added as needed

## 🤝 Contributing

To contribute to this project:

1. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is currently private. Contact the maintainer for licensing information.

## 👨‍💼 Author

**Shervin Ghaffari**
- GitHub: [@shervinghaffari79](https://github.com/shervinghaffari79)
- Email: shervinghaffari79@gmail.com

## 📞 Support

For issues, questions, or suggestions, please [open an issue](https://github.com/shervinghaffari79/STT_Persian/issues) on GitHub.

---

**Happy coding! 🚀**
