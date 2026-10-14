#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --extra web
[[ -d frontend/node_modules ]] || npm --prefix frontend ci

# Stop the whole process group, including npm's vite child, on any exit
trap 'trap - EXIT INT TERM; kill 0' EXIT INT TERM

uv run --extra web uvicorn coolfrog.api.main:app --host 127.0.0.1 --port 8010 --reload --reload-dir src &
api=$!
until (echo > /dev/tcp/127.0.0.1/8010) 2>/dev/null; do
  kill -0 "$api" 2>/dev/null || { echo 'API failed to start.' >&2; exit 1; }
  sleep 1
done
# Host, port and strictPort come from frontend/package.json and vite.config.ts
npm --prefix frontend run dev &
echo 'Coolfrog: http://127.0.0.1:5173'
wait -n
