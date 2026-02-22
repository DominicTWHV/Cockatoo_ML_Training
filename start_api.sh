#!/bin/bash
# Start the inference API server with Hypercorn

# Configuration
HOST=${1:-0.0.0.0}
PORT=${2:-8000}
WORKERS=${3:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Constellation One Text Inference API..."
echo "  Server: Quart/Hypercorn"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo ""
echo "API Endpoints:"
echo "  GET  /health          - Health check"
echo "  POST /predict         - Single inference"
echo "  POST /batch           - Batch inference"
echo ""

export HOST PORT WORKERS

screen -dmS inference bash -lc "cd $PROJECT_DIR && source venv/bin/activate && hypercorn app:app --bind \"$HOST:$PORT\" --workers \"$WORKERS\""
screen -r inference