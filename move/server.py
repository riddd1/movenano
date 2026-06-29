#!/usr/bin/env python3
"""
server.py — local web server for the handheld.py UI.
Run:  python3 server.py
Then open http://localhost:5000
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Auto-install Flask if missing
try:
    from flask import Flask, jsonify, request, send_file
except ImportError:
    print("Flask not found — installing…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "flask"],
        stdout=subprocess.DEVNULL,
    )
    from flask import Flask, jsonify, request, send_file

BASE_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE_DIR))

# In-memory job store: job_id → {status, progress, output_path, error}
_jobs: dict = {}
_lock = threading.Lock()


def _run_job(job_id: str, input_path: str, output_path: str, params: dict) -> None:
    cmd = [
        sys.executable,
        str(BASE_DIR / "handheld.py"),
        "--input",     input_path,
        "--output",    output_path,
        "--duration",  str(params["duration"]),
        "--intensity", str(params["intensity"]),
        "--seed",      str(params["seed"]),
        "--fps",       str(params["fps"]),
        "--direction", params["direction"],
        "--speed",     str(params["speed"]),
    ]
    if params["crop"]:
        cmd += ["--crop"] + [str(v) for v in params["crop"]]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
        )
        log = []
        for line in proc.stdout:
            line = line.rstrip()
            log.append(line)
            # Parse "frame X/Y" progress lines emitted by handheld.py
            m = re.search(r"frame\s+(\d+)/(\d+)", line)
            if m:
                pct = int(100 * int(m.group(1)) / int(m.group(2)))
                with _lock:
                    _jobs[job_id]["progress"] = min(pct, 99)

        proc.wait()
        with _lock:
            if proc.returncode == 0:
                _jobs[job_id].update({"status": "done", "progress": 100})
            else:
                _jobs[job_id].update({
                    "status": "error",
                    "error": "\n".join(log[-6:]) or "handheld.py exited non-zero",
                })
    except Exception as exc:
        with _lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


@app.route("/")
def index():
    return send_file(str(BASE_DIR / "index.html"))


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    params = {
        "duration":  float(request.form.get("duration",  6)),
        "intensity": float(request.form.get("intensity", 1.0)),
        "seed":      int(request.form.get("seed",        42)),
        "fps":       float(request.form.get("fps",       30)),
        "direction": request.form.get("direction", "balanced"),
        "speed":     float(request.form.get("speed", 5)),
        "crop":      None,
    }
    cx = request.form.get("crop_x", type=int)
    cy = request.form.get("crop_y", type=int)
    cw = request.form.get("crop_w", type=int)
    ch = request.form.get("crop_h", type=int)
    if all(v is not None for v in (cx, cy, cw, ch)):
        params["crop"] = [cx, cy, cw, ch]

    job_id  = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"handheld_{job_id[:8]}_")
    ext     = os.path.splitext(file.filename)[1] or ".png"
    in_path = os.path.join(tmp_dir, f"input{ext}")
    out_path = os.path.join(tmp_dir, "output.mp4")
    file.save(in_path)

    with _lock:
        _jobs[job_id] = {
            "status":      "running",
            "progress":    0,
            "output_path": out_path,
            "error":       None,
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, in_path, out_path, params),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "error":    job["error"],
    })


@app.route("/preview/<job_id>")
def preview(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 400
    return send_file(str(job["output_path"]), mimetype="video/mp4")


@app.route("/download/<job_id>")
def download(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 400
    return send_file(
        str(job["output_path"]),
        mimetype="video/mp4",
        as_attachment=True,
        download_name="handheld_output.mp4",
    )


if __name__ == "__main__":
    print("Open http://localhost:8080 in your browser")
    app.run(port=8080, debug=False, threaded=True)
