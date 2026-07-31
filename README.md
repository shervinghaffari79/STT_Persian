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
- ⏱️ **Timestamped segments** with per-word timings; export to SRT / TXT / JSON
- 🤖 **AI analysis panel** — chat over the transcript with a **local Qwen3-4B (MLX)** model (streamed, on-device, no cloud)
- 🎨 **Modern, responsive dark UI** (React + Tailwind + Vite)

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
  `../models/whisper-large-v3-persian-mlx-q8` (repo root).
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
| `POST` | `/api/chat` | `{ messages, transcript }` → streamed Persian reply (local Qwen3-4B) |
| `POST` | `/api/chat/title` | `{ transcript }` → `{ title }` |
| `GET` | `/api/health` | model presence check (`gpt_correct_available` reflects whether `OPENAI_API_KEY` is set) |

The ASR (Whisper) and chat (Qwen3-4B) models run entirely locally — on MLX/Metal
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
| Chat | MLX (`mlx-lm`) | `transformers` (CUDA if present, else CPU) |

### 1. Get the models onto the server
- ASR: `models/whisper-large-v3-persian-ct2-int8/` (CTranslate2 int8 Whisper large-v3
  fine-tuned for Persian) — copy this directory from the repo root, or re-download
  via `huggingface-cli download` if you have the original model id.
- Chat: `Qwen/Qwen3-4B-Instruct-2507` is fetched automatically from Hugging Face
  the first time `backend/chat.py` runs (no manual step, just needs the HF cache
  to have internet access once).

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
  `$env:FRONTEND_PORT` (default 5000), set before running `.\run.ps1`.
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
  accepted (`huggingface-cli login`, then visit
  https://hf.co/pyannote/speaker-diarization-3.1 and accept). Without it, the
  backend falls back to a lighter local speaker-clustering method automatically.
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
├── components/     # Reusable React components
├── pages/         # Page components
├── styles/        # Global styles
└── App.tsx        # Main application component
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
