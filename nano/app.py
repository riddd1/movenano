import base64
import os
import random
import re
import time
import uuid

from dotenv import load_dotenv, set_key
from flask import Flask, jsonify, request, send_from_directory

from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL = "gemini-2.5-flash-image"
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
MAX_ATTEMPTS = 4  # total tries per generation before giving up

# Substrings that mark a transient/overload error worth retrying.
RETRYABLE_HINTS = (
    "500", "502", "503", "429",
    "internal", "pipeline", "unavailable", "overloaded",
    "deadline", "timeout", "try again", "resource exhausted", "rate",
    # connection-level failures (e.g. parallel uploads dropping the socket)
    "broken pipe", "errno 32", "connection reset", "connection aborted",
    "connection error", "connection refused", "remote end closed",
    "eof occurred", "ssl", "read timed out", "protocolerror", "socket",
    # DNS / name-resolution blips
    "nodename nor servname", "name or service not known", "errno 8",
    "temporary failure in name resolution", "getaddrinfo", "name resolution",
    "failed to resolve", "errno -2", "errno -3", "max retries",
)


def is_retryable(message):
    msg = (message or "").lower()
    return any(hint in msg for hint in RETRYABLE_HINTS)


def backoff_seconds(attempt):
    # exponential backoff with jitter: ~0.6s, 1.2s, 2.4s, ...
    return min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0, 0.4)


def extract_image_part(response):
    try:
        for part in response.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.mime_type and inline.mime_type.startswith("image/"):
                return inline
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def no_image_reason(response):
    """Build a helpful message when the model returns no image."""
    try:
        cand = response.candidates[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        if "SAFETY" in finish.upper() or "BLOCK" in finish.upper():
            return "The request was blocked by safety filters. Try a different image or prompt."
        # Sometimes the model replies with text instead of an image.
        for part in cand.content.parts:
            txt = getattr(part, "text", None)
            if txt:
                return f"The model returned text instead of an image: {txt.strip()[:200]}"
    except (AttributeError, IndexError, TypeError):
        pass
    return "The model did not return an image (transient issue). Try generating again."

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Make sure a .env file exists so we can persist the key into it later.
if not os.path.exists(ENV_PATH):
    open(ENV_PATH, "a").close()

load_dotenv(ENV_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def get_api_key():
    return os.environ.get("GOOGLE_API_KEY", "").strip()


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Image is too large. Maximum upload size is 20MB."}), 413


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "templates"), "index.html")


@app.route("/api/status")
def status():
    return jsonify({"configured": bool(get_api_key())})


@app.route("/api/save-key", methods=["POST"])
def save_key():
    data = request.get_json(silent=True) or {}
    key = (data.get("api_key") or "").strip()
    if not key:
        return jsonify({"error": "Please enter a valid API key."}), 400
    try:
        set_key(ENV_PATH, "GOOGLE_API_KEY", key)
        os.environ["GOOGLE_API_KEY"] = key
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not save API key: {exc}"}), 500
    return jsonify({"configured": True})


@app.route("/api/generate", methods=["POST"])
def generate():
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "No API key configured. Please add your Gemini API key first."}), 400

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "Please upload an image."}), 400

    prompt_text = (request.form.get("prompt") or "").strip()
    if not prompt_text:
        return jsonify({"error": "Please enter a transformation prompt."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    mime_type = image_file.mimetype or "image/png"
    if not mime_type.startswith("image/"):
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    client = genai.Client(api_key=api_key)
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt_text,
    ]
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    image_part = None
    last_error = "The model did not return an image."
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"The Gemini API request failed: {exc}"
            if is_retryable(str(exc)) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(backoff_seconds(attempt))
                continue
            return jsonify({"error": last_error}), 502

        image_part = extract_image_part(response)
        if image_part is not None:
            break

        # No image came back. Retry transient cases; otherwise stop early.
        last_error = no_image_reason(response)
        if "blocked by safety" in last_error or "returned text instead" in last_error:
            return jsonify({"error": last_error}), 502
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(backoff_seconds(attempt))

    if image_part is None:
        return jsonify({"error": last_error}), 502

    out_mime = image_part.mime_type
    ext = (out_mime.split("/")[-1] or "png").split(";")[0]
    ext = re.sub(r"[^a-zA-Z0-9]", "", ext) or "png"
    filename = f"{uuid.uuid4().hex}.{ext}"
    out_path = os.path.join(OUTPUT_DIR, filename)

    out_bytes = image_part.data
    with open(out_path, "wb") as fh:
        fh.write(out_bytes)

    b64 = base64.b64encode(out_bytes).decode("ascii")
    return jsonify(
        {
            "image": f"data:{out_mime};base64,{b64}",
            "download_url": f"/outputs/{filename}",
            "filename": filename,
        }
    )


@app.route("/outputs/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
