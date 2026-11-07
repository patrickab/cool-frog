#!/usr/bin/env bash
# Usage:
#   ./run-frontend.sh             dataset explorer with its API
#   ./run-frontend.sh --monitor   training monitor, SSH hosts are configured on the page (monitor-hosts.json)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Wait until 127.0.0.1:$1 accepts connections, exit when process $2 ($3) died first
wait_for_port() {
  until (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "$3 failed to start." >&2; exit 1; }
    sleep 1
  done
}

case ${1:-} in
  '') api=coolfrog.api.main port=8010 page= ;;
  --monitor) api=coolfrog.api.monitor_hub port=8011 page=monitor.html ;;
  *) sed -n '2,4s/^# \{0,1\}//p' "$0" >&2; exit 2 ;;
esac

uv sync
[[ -d frontend/node_modules ]] || npm --prefix frontend ci

# Stop the whole process group, including npm's vite child, on any exit
trap 'trap - EXIT INT TERM; kill 0' EXIT INT TERM

uv run uvicorn "$api:app" --host 127.0.0.1 --port "$port" --reload --reload-dir src &
wait_for_port "$port" $! API
# Host, port and strictPort come from frontend/package.json and vite.config.ts
npm --prefix frontend run dev &
wait_for_port 5173 $! Vite
# Printed after Vite's own banner, whose bare http://127.0.0.1:5173/ is the explorer
echo "Coolfrog: http://127.0.0.1:5173/$page"
wait -n
