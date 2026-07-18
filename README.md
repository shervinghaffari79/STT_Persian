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

## ✨ Features

- 🎯 **Local Persian ASR** — fine-tuned Whisper large-v3, GPU-accelerated via MLX
- 🗣️ **Speaker diarization** — automatic speaker separation and labels
- ⏱️ **Timestamped segments** with per-word timings; export to SRT / TXT / JSON
- 🤖 **AI analysis panel** — chat over the transcript with a **local Qwen3-4B (MLX)** model (streamed, on-device, no cloud)
- 🎨 **Modern, responsive dark UI** (React + Tailwind + Vite)

## 🏗️ Architecture

```
Browser (React/Vite :5173)
   │  POST /api/transcribe  (multipart upload)
   │  GET  /api/status/{id} (poll progress)   ── Vite proxy ──▶  FastAPI :8000
   │                                                              backend/server.py
   ▼                                                              backend/pipeline.py
Transcript + speakers rendered in the middle panel
```

## 🚀 Getting Started

### Prerequisites
- **Node.js 16+** and npm
- **Python 3.9+** and **ffmpeg** (`brew install ffmpeg`)
- Apple Silicon recommended (MLX uses the Metal GPU; falls back to CPU elsewhere)
- The MLX model at `../models/whisper-large-v3-persian-mlx-q8` (repo root)

### Run everything (backend + frontend)

```bash
# one-time: install deps
pip3 install -r backend/requirements.txt
npm install

# start backend (:8000) AND frontend (:5173) together
./run.sh
```

Then open **http://localhost:5173**, drop an audio/`.mp4` file, and click **Transcribe Audio**.

### Or run the two services separately

```bash
# terminal 1 — backend
cd backend && python3 server.py         # FastAPI on http://127.0.0.1:8000

# terminal 2 — frontend
npm run dev                              # Vite on http://localhost:5173
```

The frontend proxies `/api/*` to the backend (see `vite.config.ts`).

### Backend API
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/transcribe` | multipart `file` (+ `diarize=true\|false`) → `{ job_id }` |
| `GET` | `/api/status/{job_id}` | `{ state, progress, message, result? }` |
| `POST` | `/api/chat` | `{ messages, transcript }` → streamed Persian reply (local Qwen3-4B MLX) |
| `POST` | `/api/chat/title` | `{ transcript }` → `{ title }` |
| `GET` | `/api/health` | model presence check |

Both the ASR (Whisper) and chat (Qwen3-4B) models run locally via MLX on the
Apple-Silicon GPU. Nothing is sent to any external API.

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
   The application will be available at `http://localhost:5173`

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
