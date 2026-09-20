#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

api_pid=""
web_pid=""

cleanup() {
  trap - TERM INT EXIT
  echo
  echo "Stopping ScopeGraph API and web UI..."
  [[ -z "$api_pid" ]] || kill "$api_pid" 2>/dev/null || true
  [[ -z "$web_pid" ]] || kill "$web_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

command -v uv >/dev/null || { echo "Error: uv is required." >&2; exit 1; }
command -v npm >/dev/null || { echo "Error: npm is required." >&2; exit 1; }
command -v docker >/dev/null || { echo "Error: Docker is required." >&2; exit 1; }

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your provider credentials before using live features."
fi

echo "Installing backend dependencies..."
uv sync --extra dev

echo "Installing web dependencies..."
npm --prefix web install

echo "Starting Neo4j..."
docker compose up -d neo4j

echo "Waiting for Neo4j..."
for attempt in {1..30}; do
  if curl -fsS http://127.0.0.1:7474 >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" == 30 ]]; then
    echo "Error: Neo4j did not become ready." >&2
    exit 1
  fi
  sleep 2
done

echo "Applying Neo4j schema..."
PYTHONPATH=src:. uv run python scripts/setup_neo4j.py

echo "Starting API at http://127.0.0.1:8000"
PYTHONPATH=src:. uv run uvicorn scopegraph.api.main:app --reload >"$ROOT_DIR/.scopegraph-api.log" 2>&1 &
api_pid=$!

echo "Starting web UI at http://localhost:5173"
npm --prefix web run dev >"$ROOT_DIR/.scopegraph-web.log" 2>&1 &
web_pid=$!

echo
echo "ScopeGraph is running. Press Ctrl+C to stop the API and web UI."
echo "API docs: http://127.0.0.1:8000/docs"
echo "Logs: .scopegraph-api.log and .scopegraph-web.log"

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
  sleep 2
done

echo "A development process exited. Check .scopegraph-api.log or .scopegraph-web.log." >&2
exit 1
