import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid

# Resolve a real ffmpeg binary up front instead of hoping "ffmpeg" is on PATH —
# system ffmpeg isn't guaranteed to exist on every deploy target (e.g. Railway's
# Railpack builder doesn't honor nixpacks.toml's aptPkgs), so imageio-ffmpeg's
# bundled/downloaded binary is the reliable fallback. Any failure here is logged
# instead of silently swallowed, so a bad deploy shows up in the logs immediately.
FFMPEG_BIN = "ffmpeg"
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as _exc:
    _which = shutil.which("ffmpeg")
    if _which:
        FFMPEG_BIN = _which
    else:
        print(f"[startup] Could not resolve an ffmpeg binary via imageio-ffmpeg ({_exc}) "
              f"and none found on PATH; video features will fail until this is fixed.",
              file=sys.stderr)
os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, jsonify, request, send_from_directory, send_file

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MOVE_DIR   = os.path.join(BASE_DIR, "move")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HANDHELD   = os.path.join(MOVE_DIR, "handheld.py")
MAX_SIZE    = 500 * 1024 * 1024  # generous cap so TL can upload several video clips at once
CONCAT_TIMEOUT = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE

_jobs: dict = {}
_lock = threading.Lock()


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Upload too large."}), 413


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


# ── TL routes ─────────────────────────────────────────────────

def _probe_video_dims(path):
    """Read WxH from ffmpeg's stderr banner — avoids depending on a separate ffprobe binary."""
    proc = subprocess.run([FFMPEG_BIN, "-i", path], capture_output=True, text=True, timeout=20)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", proc.stderr)
    if not m:
        raise ValueError(f"Could not read video dimensions for {os.path.basename(path)}")
    return int(m.group(1)), int(m.group(2))


@app.route("/api/concat", methods=["POST"])
def api_concat():
    clips = request.files.getlist("clips")
    if not clips:
        return jsonify({"error": "No clips provided."}), 400

    # Optional per-clip trim durations (seconds), same order as `clips`. An empty
    # string / missing entry / non-positive value means "don't trim this one".
    raw_durations = request.form.getlist("durations")
    durations = []
    for i in range(len(clips)):
        raw = raw_durations[i] if i < len(raw_durations) else ""
        try:
            d = float(raw)
            durations.append(d if d > 0 else None)
        except (TypeError, ValueError):
            durations.append(None)

    tmp_dir = tempfile.mkdtemp()
    try:
        in_paths = []
        for i, f in enumerate(clips):
            ext = os.path.splitext(f.filename or "clip.webm")[1] or ".webm"
            p = os.path.join(tmp_dir, f"clip{i:03d}{ext}")
            f.save(p)
            in_paths.append(p)

        try:
            dims = [_probe_video_dims(p) for p in in_paths]
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        target_w = max(w for w, _ in dims)
        target_h = max(h for _, h in dims)
        target_w += target_w % 2
        target_h += target_h % 2

        filter_parts = []
        concat_inputs = ""
        for i in range(len(in_paths)):
            filter_parts.append(
                f"[{i}:v]scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30[v{i}]"
            )
            concat_inputs += f"[v{i}]"
        filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(in_paths)}:v=1:a=0[outv]"

        out_filename = f"{uuid.uuid4().hex}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        cmd = [FFMPEG_BIN, "-y"]
        for p, dur in zip(in_paths, durations):
            if dur is not None:
                cmd += ["-t", str(dur)]  # input-side -t: stop reading this clip after `dur` seconds
            cmd += ["-i", p]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CONCAT_TIMEOUT)
        if proc.returncode != 0:
            return jsonify({"error": f"Render failed: {proc.stderr[-800:]}"}), 500

        return jsonify({"download_url": f"/outputs/{out_filename}", "filename": out_filename})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Render timed out."}), 504
    except Exception as exc:
        return jsonify({"error": f"Render failed: {exc}"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    try:
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

        job_id   = str(uuid.uuid4())
        tmp_dir  = tempfile.mkdtemp()
        ext      = os.path.splitext(file.filename or "input.png")[1] or ".png"
        in_path  = os.path.join(tmp_dir, f"input{ext}")
        out_path = os.path.join(tmp_dir, "output.mp4")

        img_bytes = file.read()
        with open(in_path, "wb") as fh:
            fh.write(img_bytes)

        with _lock:
            _jobs[job_id] = {"status": "running", "progress": 0, "output_path": out_path, "error": None}

        threading.Thread(target=_run_move_job, args=(job_id, in_path, out_path, params), daemon=True).start()
        return jsonify({"job_id": job_id})
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500


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
