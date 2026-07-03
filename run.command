#!/bin/bash
# Double-click this in Finder to start Fix + Move.
cd "$(dirname "$0")"

# Reuse nano's virtual environment (already has all dependencies).
VENV="nano/venv"

if [ ! -d "$VENV" ]; then
  echo "First-time setup: creating virtual environment..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -r nano/requirements.txt
fi

# Install Move deps if missing
if ! "$VENV/bin/python" -c "import cv2" 2>/dev/null; then
  echo "Installing Move dependencies (opencv-python, numpy)..."
  "$VENV/bin/pip" install opencv-python numpy
fi

echo ""
echo "Fix + Move is starting..."
echo "Open this in your browser:  http://localhost:5002"
echo "(Leave this window open. Press Ctrl+C to stop.)"
echo ""

"$VENV/bin/python" app.py
