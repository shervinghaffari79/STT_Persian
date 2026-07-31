#!/usr/bin/env bash
# Launch the local Persian SOTA ASR app.
#
# Only the FRONTEND (Vite) is network/internet-exposed, on 0.0.0.0:5000
# (override with FRONTEND_PORT). The FastAPI backend stays on 127.0.0.1:8000
# (override with BACKEND_PORT) and is reached only through the frontend's
# built-in proxy -- it is never reachable directly from outside this machine.
# There is currently NO authentication on the backend's API, so put a
# firewall/VPN/security-group rule in front of :5000 before exposing it
# beyond a trusted network.
#
# Usage: ./run.sh   (Ctrl-C stops both)
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5000}"

echo "▶ starting backend (FastAPI, 127.0.0.1:$BACKEND_PORT, not exposed) …"
( cd backend && HOST=127.0.0.1 PORT="$BACKEND_PORT" python3 server.py ) &
BACK=$!

cleanup() { echo; echo "stopping…"; kill "$BACK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# wait for backend health
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
    echo "✓ backend ready"; break
  fi
  sleep 1
done

echo "▶ starting frontend (Vite, 0.0.0.0:$FRONTEND_PORT, network-exposed) …"
BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" npm run dev
