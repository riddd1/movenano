import base64
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from dotenv import load_dotenv, set_key
from flask import Flask, jsonify, request, send_from_directory, send_file

from google import genai
from google.genai import types

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NANO_DIR   = os.path.join(BASE_DIR, "nano")
MOVE_DIR   = os.path.join(BASE_DIR, "move")
ENV_PATH   = os.path.join(BASE_DIR, ".env")  # root-level .env for Railway compat
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HANDHELD   = os.path.join(MOVE_DIR, "handheld.py")
MODEL      = "gemini-2.5-flash-image"
MAX_SIZE   = 20 * 1024 * 1024
MAX_TRIES  = 4

RETRYABLE = (
    "500","502","503","429","internal","pipeline","unavailable","overloaded",
    "deadline","timeout","try again","resource exhausted","rate","broken pipe",
    "errno 32","connection reset","connection aborted","connection error",
    "connection refused","remote end closed","eof occurred","ssl","read timed out",
    "protocolerror","socket","nodename nor servname","name or service not known",
    "errno 8","temporary failure in name resolution","getaddrinfo","name resolution",
    "failed to resolve","errno -2","errno -3","max retries",
)

def is_retryable(msg):
    m = (msg or "").lower()
    return any(h in m for h in RETRYABLE)

def backoff(attempt):
    return min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0, 0.4)

def extract_image(response):
    try:
        for part in response.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.mime_type and inline.mime_type.startswith("image/"):
                return inline
    except (AttributeError, IndexError, TypeError):
        pass
    return None

def no_image_reason(response):
    try:
        cand = response.candidates[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        if "SAFETY" in finish.upper() or "BLOCK" in finish.upper():
            return "Request blocked by safety filters."
        for part in cand.content.parts:
            txt = getattr(part, "text", None)
            if txt:
                return f"Model returned text instead of an image: {txt.strip()[:200]}"
    except (AttributeError, IndexError, TypeError):
        pass
    return "Model did not return an image. Try again."

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(ENV_PATH):
    open(ENV_PATH, "a").close()
load_dotenv(ENV_PATH)  # local .env; Railway env vars take precedence automatically

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE

_jobs: dict = {}
_lock = threading.Lock()


def get_key():
    return os.environ.get("GOOGLE_API_KEY", "").strip()


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Image too large. Max 20MB."}), 413


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))


# ── Nano routes ──────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({"configured": bool(get_key())})


@app.route("/api/save-key", methods=["POST"])
def api_save_key():
    data = request.get_json(silent=True) or {}
    key  = (data.get("api_key") or "").strip()
    if not key:
        return jsonify({"error": "Please enter a valid API key."}), 400
    try:
        set_key(ENV_PATH, "GOOGLE_API_KEY", key)
        os.environ["GOOGLE_API_KEY"] = key
    except Exception as exc:
        return jsonify({"error": f"Could not save key: {exc}"}), 500
    return jsonify({"configured": True})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    key = get_key()
    if not key:
        return jsonify({"error": "No API key. Add your Gemini API key first."}), 400
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "Please upload an image."}), 400
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Please enter a prompt."}), 400

    img_file  = request.files["image"]
    img_bytes = img_file.read()
    mime      = img_file.mimetype or "image/png"
    if not mime.startswith("image/"):
        return jsonify({"error": "Not a valid image file."}), 400

    client   = genai.Client(api_key=key)
    contents = [types.Part.from_bytes(data=img_bytes, mime_type=mime), prompt]
    config   = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    img_part   = None
    last_error = "Model did not return an image."
    for attempt in range(MAX_TRIES):
        try:
            resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        except Exception as exc:
            last_error = f"Gemini API error: {exc}"
            if is_retryable(str(exc)) and attempt < MAX_TRIES - 1:
                time.sleep(backoff(attempt))
                continue
            return jsonify({"error": last_error}), 502

        img_part = extract_image(resp)
        if img_part:
            break
        last_error = no_image_reason(resp)
        if "safety" in last_error or "text instead" in last_error:
            return jsonify({"error": last_error}), 502
        if attempt < MAX_TRIES - 1:
            time.sleep(backoff(attempt))

    if not img_part:
        return jsonify({"error": last_error}), 502

    out_mime = img_part.mime_type
    ext      = re.sub(r"[^a-zA-Z0-9]", "", (out_mime.split("/")[-1] or "png").split(";")[0]) or "png"
    filename = f"{uuid.uuid4().hex}.{ext}"
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "wb") as fh:
        fh.write(img_part.data)

    b64 = base64.b64encode(img_part.data).decode("ascii")
    return jsonify({
        "image":        f"data:{out_mime};base64,{b64}",
        "download_url": f"/outputs/{filename}",
        "filename":     filename,
    })


@app.route("/outputs/<path:filename>")
def nano_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


# ── Move routes ───────────────────────────────────────────────

def _run_move_job(job_id, in_path, out_path, params):
    cmd = [
        sys.executable, HANDHELD,
        "--input",     in_path,
        "--output",    out_path,
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=MOVE_DIR,
        )
        log = []
        for line in proc.stdout:
            line = line.rstrip()
            log.append(line)
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
                    "error":  "\n".join(log[-6:]) or "handheld.py failed",
                })
    except Exception as exc:
        with _lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


@app.route("/move/generate", methods=["POST"])
def move_generate():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    params = {
        "duration":  float(request.form.get("duration",  6)),
        "intensity": float(request.form.get("intensity", 1.0)),
        "seed":      int(request.form.get("seed",        42)),
        "fps":       float(request.form.get("fps",       30)),
        "direction": request.form.get("direction", "balanced"),
        "speed":     float(request.form.get("speed",    5)),
        "crop":      None,
    }
    cx, cy = request.form.get("crop_x", type=int), request.form.get("crop_y", type=int)
    cw, ch = request.form.get("crop_w", type=int), request.form.get("crop_h", type=int)
    if all(v is not None for v in (cx, cy, cw, ch)):
        params["crop"] = [cx, cy, cw, ch]

    job_id  = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"handheld_{job_id[:8]}_")
    ext     = os.path.splitext(file.filename)[1] or ".png"
    in_path = os.path.join(tmp_dir, f"input{ext}")
    out_path = os.path.join(tmp_dir, "output.mp4")
    file.save(in_path)

    with _lock:
        _jobs[job_id] = {"status": "running", "progress": 0, "output_path": out_path, "error": None}

    threading.Thread(target=_run_move_job, args=(job_id, in_path, out_path, params), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/move/status/<job_id>")
def move_status(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"], "error": job["error"]})


@app.route("/move/preview/<job_id>")
def move_preview(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 400
    return send_file(job["output_path"], mimetype="video/mp4")


@app.route("/move/download/<job_id>")
def move_download(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 400
    return send_file(job["output_path"], mimetype="video/mp4", as_attachment=True, download_name="handheld.mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    print(f"\nOpen http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
