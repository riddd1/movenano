#!/bin/bash
# Double-click this file in Finder to start Nano Banana.
cd "$(dirname "$0")"

# Create the virtual environment + install deps the first time only.
if [ ! -d "venv" ]; then
  echo "First-time setup: creating virtual environment and installing dependencies..."
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
fi

echo ""
echo "🍌 Nano Banana is starting..."
echo "Open this in your browser:  http://localhost:5001"
echo "(Leave this window open. Press Ctrl+C here to stop.)"
echo ""

./venv/bin/python app.py
