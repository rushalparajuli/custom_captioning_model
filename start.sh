#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate Python virtual environment and start Flask app in background
source venv/bin/activate
python3 app.py &
PYTHON_PID=$!

# Start Go server in background
go run main.go &
GO_PID=$!

# When script exits (e.g. Ctrl+C), stop Python and Go
trap "kill $PYTHON_PID $GO_PID 2>/dev/null" EXIT

# Start React app in foreground
cd caption-app
npm run start
