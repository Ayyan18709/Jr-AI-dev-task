#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  –  Unified launch script for the RAG Chatbot API
# Usage:
#   ./run.sh              → run locally (virtualenv)
#   ./run.sh docker       → build & run via Docker Compose
#   ./run.sh docker-stop  → stop Docker Compose stack
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-local}"

banner() {
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║          RAG Chatbot API – Qwen2.5-1.5B          ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""
}

# ── Local run ──────────────────────────────────────────────────────────────────
run_local() {
  banner
  echo "► Mode: LOCAL"

  # Create .env if missing
  if [ ! -f ".env" ]; then
    echo "  ⚠  .env not found – copying from .env.example"
    cp .env.example .env
  fi

  # Virtualenv
  if [ ! -d "venv" ]; then
    echo "  Creating virtualenv …"
    python -m venv venv
  fi

  source venv/bin/activate || source venv/Scripts/activate 2>/dev/null

  echo "  Installing dependencies …"
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt

  # Create required dirs
  mkdir -p data/uploads data/faiss_index logs

  echo ""
  echo "  Starting API on http://localhost:8000"
  echo "  Docs: http://localhost:8000/docs"
  echo ""
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# ── Docker run ─────────────────────────────────────────────────────────────────
run_docker() {
  banner
  echo "► Mode: DOCKER"

  if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ⚠  Created .env from .env.example – edit as needed."
  fi

  echo "  Building image and starting services …"
  docker compose -f docker/docker-compose.yml up --build -d

  echo ""
  echo "  Services started:"
  echo "    API   → http://localhost:8000"
  echo "    Docs  → http://localhost:8000/docs"
  echo "    Redis → localhost:6379"
  echo ""
  echo "  Logs: docker compose -f docker/docker-compose.yml logs -f api"
  echo "  Stop: ./run.sh docker-stop"
}

# ── Docker stop ────────────────────────────────────────────────────────────────
stop_docker() {
  echo "  Stopping Docker Compose stack …"
  docker compose -f docker/docker-compose.yml down
  echo "  Done."
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "$MODE" in
  local)        run_local ;;
  docker)       run_docker ;;
  docker-stop)  stop_docker ;;
  *)
    echo "Usage: $0 [local|docker|docker-stop]"
    exit 1
    ;;
esac
