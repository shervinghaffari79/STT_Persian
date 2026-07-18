#!/usr/bin/env bash
# Launch the local Persian SOTA ASR app: FastAPI backend + Vite frontend.
# Usage: ./run.sh   (Ctrl-C stops both)
set -euo pipefail
cd "$(dirname "$0")"

echo "▶ starting backend (FastAPI :8000) …"
( cd backend && python3 server.py ) &
BACK=$!

cleanup() { echo; echo "stopping…"; kill "$BACK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# wait for backend health
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "✓ backend ready"; break
  fi
  sleep 1
done

echo "▶ starting frontend (Vite :5173) …"
npm run dev
