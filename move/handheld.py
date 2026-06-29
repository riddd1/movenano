#!/usr/bin/env python3
"""
handheld.py — Apply subtle, realistic handheld camera motion to a video or image.

Dependencies: opencv-python, numpy, ffmpeg + ffprobe (both on system PATH)

Usage:
    # Apply motion to an existing video:
    python handheld.py --input clip.mp4 --output out.mp4 [--intensity 1.0] [--seed 42]

    # Animate a static image into a clip:
    python handheld.py --input photo.jpg --output out.mp4 \
                       [--intensity 1.0] [--seed 42] [--fps 30] [--duration 6]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
#  MOTION PARAMETERS
#  All amplitude values scale linearly with --intensity.
#  Edit this block to change the feel; leave the code below untouched.
# ============================================================
P = {
    # 1. Band-limited smooth drift
    #    Summing random low-frequency sinusoids gives smooth, organic-feeling noise.
    "drift_amp":      2.5,    # px — peak amplitude per axis at intensity=1
    "drift_max_freq": 0.80,   # Hz — raised from 0.4 so direction changes are less predictable
    "drift_n_comp":   12,     # number of sinusoidal terms in the sum

    # 2. Lissajous figure-eight overlay — kept subtle so it doesn't create obvious oscillation
    #    tx = A·sin(2π·f·t),  ty = B·sin(2π·f·ky·t + φ)
    "liss_freq":      0.07,   # Hz base frequency (~14 s per full loop)
    "liss_amp_x":     0.6,    # px — reduced so drift/tremor dominate; avoids back-and-forth look
    "liss_amp_y":     0.5,    # px
    "liss_ky":        2.03,   # y-frequency multiplier — slight detuning avoids exact repeat

    # 3. Breathing sway — slow vertical sine mimicking natural shoulder movement
    "breathe_amp":    1.5,    # px
    "breathe_period": 6.2,    # seconds

    # 4. Ornstein–Uhlenbeck micro-corrections (x and y independently)
    #    Models a person constantly making tiny subconscious stabilisation adjustments.
    "ou_theta_xy":    1.2,    # mean-reversion rate s⁻¹ (higher → faster pull-back)
    "ou_sigma_xy":    0.90,   # noise intensity px/√s — raised for more erratic micro-moves

    # 5. Rotation / tilt (smooth noise + OU)
    "rot_amp":        0.20,   # degrees — peak amplitude of smooth tilt noise
    "rot_max_freq":   0.25,   # Hz
    "rot_n_comp":     8,
    "ou_theta_rot":   0.9,    # OU mean-reversion rate for rotation
    "ou_sigma_rot":   0.006,  # degrees/√s for rotation OU component

    # 6. Zoom variation — hand drifting slightly toward/away from the scene
    "zoom_amp":       0.007,  # ±0.7% scale — subtle forward/back movement
    "zoom_max_freq":  0.18,   # Hz — slow oscillation (~5-6 s per cycle)
    "zoom_n_comp":    6,

    # 7. Physiological tremor — muscle micro-shake that bypasses the low-pass smoother
    #    This is what gives handheld footage its nervous, human quality.
    "tremor_amp":      0.70,  # px peak — raised for more noticeable organic texture
    "tremor_min_freq": 4.0,   # Hz — lower bound of natural hand tremor
    "tremor_max_freq": 9.0,   # Hz — upper bound (Nyquist-safe at 30 fps)
    "tremor_n_comp":   10,    # more components = more irregular tremor texture

    # Final low-pass smoothing of the SLOW components only (tremor is added after)
    "smooth_sigma":   0.9,    # Gaussian sigma in frames — kept low for rough, human feel

    # Minimum safety zoom-in factor (auto-expanded from actual peak displacement)
    "min_zoom":       1.018,
}


# ============================================================
#  SIGNAL GENERATORS
# ============================================================

def band_limited_noise(n: int, fps: float, amplitude: float,
                       max_freq: float, n_comp: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Band-limited smooth noise: sum of n_comp sinusoids at random frequencies
    in [0.005, max_freq] Hz with random phases and weights.
    The result is normalised so its peak absolute value equals `amplitude`.
    """
    t   = np.arange(n) / fps
    sig = np.zeros(n)
    for _ in range(n_comp):
        freq   = rng.uniform(0.005, max_freq)
        phase  = rng.uniform(0.0, 2.0 * math.pi)
        weight = rng.uniform(0.5, 1.0)
        sig   += weight * np.sin(2.0 * math.pi * freq * t + phase)
    peak = np.max(np.abs(sig))
    if peak > 1e-12:
        sig *= amplitude / peak
    return sig


def ou_process(n: int, fps: float, theta: float, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Discrete Ornstein–Uhlenbeck process (Euler–Maruyama):
        x[t+dt] = x[t] − θ·x[t]·dt + σ·√dt·N(0,1)

    Produces a mean-reverting walk toward zero — the mathematical model of a
    person trying to hold a camera still: any drift is continuously corrected,
    but each correction overshoots slightly and must itself be corrected.
    """
    dt   = 1.0 / fps
    sqdt = math.sqrt(dt)
    x    = np.zeros(n)
    dw   = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] * dt + sigma * sqdt * dw[i]
    return x


def gaussian_smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase 1-D Gaussian smoothing via edge-padded numpy convolution."""
    if sigma_frames <= 0.0:
        return x.copy()
    r      = max(1, int(math.ceil(3.0 * sigma_frames)))
    k      = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-0.5 * (k / sigma_frames) ** 2)
    kernel /= kernel.sum()
    padded   = np.pad(x, r, mode="edge")   # edge-pad reduces boundary ringing
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(x)]


# ============================================================
#  MOTION CURVE BUILDER
# ============================================================

# Axis-bias scales: (x_multiplier, y_multiplier)
_DIRECTION_SCALES = {
    "balanced":   (1.00, 1.00),
    "horizontal": (1.70, 0.45),   # dominant left-right drift
    "vertical":   (0.45, 1.70),   # dominant up-down drift
}


def build_motion(n_frames: int, fps: float,
                 intensity: float, seed: int,
                 direction: str = "balanced",
                 speed: float = 5.0,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tx, ty, theta, zoom_var): per-frame translation (px), tilt (deg),
    and zoom variation (fractional scale delta, centred at 0).
    `direction` biases the shake axis; `speed` (1–10) scales all motion frequencies.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_frames) / fps

    # Speed 1–10 → frequency multiplier centred at 1.0 when speed=5
    sp = max(0.1, speed / 5.0)
    drift_max_freq  = P["drift_max_freq"]  * sp
    liss_freq       = P["liss_freq"]       * sp
    breathe_period  = P["breathe_period"]  / sp          # period is inverse of frequency
    ou_theta_xy     = P["ou_theta_xy"]     * sp
    ou_theta_rot    = P["ou_theta_rot"]    * sp
    smooth_sigma    = P["smooth_sigma"]    / sp          # less smoothing at high speed = rougher

    # --- 1. Band-limited smooth drift ---
    tx_drift = band_limited_noise(n_frames, fps, P["drift_amp"] * intensity,
                                   drift_max_freq, P["drift_n_comp"], rng)
    ty_drift = band_limited_noise(n_frames, fps, P["drift_amp"] * intensity,
                                   drift_max_freq, P["drift_n_comp"], rng)

    # --- 2. Lissajous figure-eight ---
    ph_liss = rng.uniform(0.0, 2.0 * math.pi)
    tx_liss = P["liss_amp_x"] * intensity * np.sin(2.0 * math.pi * liss_freq * t)
    ty_liss = P["liss_amp_y"] * intensity * np.sin(
        2.0 * math.pi * liss_freq * P["liss_ky"] * t + ph_liss)

    # --- 3. Breathing sway (vertical only) ---
    ph_breathe = rng.uniform(0.0, 2.0 * math.pi)
    ty_breathe = (P["breathe_amp"] * intensity *
                  np.sin(2.0 * math.pi * t / breathe_period + ph_breathe))

    # --- 4. Ornstein–Uhlenbeck micro-corrections ---
    tx_ou = ou_process(n_frames, fps, ou_theta_xy, P["ou_sigma_xy"] * intensity, rng)
    ty_ou = ou_process(n_frames, fps, ou_theta_xy, P["ou_sigma_xy"] * intensity, rng)

    # --- 5. Rotation (smooth noise + OU combined) ---
    rot_noise = band_limited_noise(n_frames, fps, P["rot_amp"] * intensity,
                                    P["rot_max_freq"] * sp, P["rot_n_comp"], rng)
    rot_ou    = ou_process(n_frames, fps,
                            ou_theta_rot, P["ou_sigma_rot"] * intensity, rng)
    theta = rot_noise + rot_ou

    # --- 6. Zoom variation (hand drifting toward/away from scene) ---
    zoom_var = band_limited_noise(n_frames, fps, P["zoom_amp"] * intensity,
                                   P["zoom_max_freq"], P["zoom_n_comp"], rng)

    # --- Combine slow components and smooth them ---
    tx = tx_drift + tx_liss + tx_ou
    ty = ty_drift + ty_liss + ty_breathe + ty_ou

    tx       = gaussian_smooth(tx,       smooth_sigma)
    ty       = gaussian_smooth(ty,       smooth_sigma)
    theta    = gaussian_smooth(theta,    smooth_sigma)
    zoom_var = gaussian_smooth(zoom_var, smooth_sigma)

    # --- 7. Physiological tremor — added AFTER smoothing so it isn't filtered out ---
    #    High-frequency (4-9 Hz) muscle shake; this is what makes footage feel human.
    #    x and y get independent tremor so it doesn't look like a diagonal vibration.
    tremor_x = np.zeros(n_frames)
    tremor_y = np.zeros(n_frames)
    for _ in range(P["tremor_n_comp"]):
        fx = rng.uniform(P["tremor_min_freq"], P["tremor_max_freq"])
        fy = rng.uniform(P["tremor_min_freq"], P["tremor_max_freq"])
        px = rng.uniform(0.0, 2.0 * math.pi)
        py = rng.uniform(0.0, 2.0 * math.pi)
        wx = rng.uniform(0.5, 1.0)
        wy = rng.uniform(0.5, 1.0)
        tremor_x += wx * np.sin(2.0 * math.pi * fx * t + px)
        tremor_y += wy * np.sin(2.0 * math.pi * fy * t + py)
    for sig, amp in [(tremor_x, P["tremor_amp"] * intensity),
                     (tremor_y, P["tremor_amp"] * intensity)]:
        pk = np.max(np.abs(sig))
        if pk > 1e-12:
            sig *= amp / pk
    tx += tremor_x
    ty += tremor_y

    # Apply axis-direction bias — scales all x and y motion together
    x_scale, y_scale = _DIRECTION_SCALES.get(direction, (1.0, 1.0))
    tx *= x_scale
    ty *= y_scale

    return tx, ty, theta, zoom_var


# ============================================================
#  ZOOM CALCULATION
# ============================================================

def compute_zoom(tx: np.ndarray, ty: np.ndarray,
                 theta_deg: np.ndarray, zoom_var: np.ndarray,
                 w: int, h: int) -> float:
    """
    Minimum base zoom so no black borders appear across the full motion envelope,
    including headroom for the per-frame zoom variation dipping below the base.
    """
    max_tx     = float(np.max(np.abs(tx)))
    max_ty     = float(np.max(np.abs(ty)))
    max_th_rad = float(np.max(np.abs(theta_deg))) * math.pi / 180.0

    corner_r  = math.hypot(w / 2.0, h / 2.0)
    rot_bleed = corner_r * abs(math.sin(max_th_rad))

    # Extra headroom so the base zoom can safely absorb negative zoom_var values
    zoom_headroom = float(np.max(np.abs(zoom_var)))

    zoom = max(
        1.0 + 2.0 * (max_tx + rot_bleed) / w,
        1.0 + 2.0 * (max_ty + rot_bleed) / h,
        P["min_zoom"] + zoom_headroom,
    )
    return zoom


# ============================================================
#  FRAME WARPING
# ============================================================

def warp_frame(frame: np.ndarray,
               tx: float, ty: float, theta_deg: float,
               zoom: float, w: int, h: int) -> np.ndarray:
    """
    Apply zoom + rotation + translation affine warp to a single BGR frame.
    Zoom is applied about the frame centre to pad all edges uniformly.
    BORDER_REFLECT is a safety net in case the zoom estimate is tight.
    """
    cx, cy = w / 2.0, h / 2.0
    # getRotationMatrix2D: scale+rotate about (cx, cy), producing a 2×3 affine matrix
    M = cv2.getRotationMatrix2D((cx, cy), theta_deg, zoom)
    # Add translation offset into the affine's translation column
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


# ============================================================
#  FFPROBE HELPER
# ============================================================

def probe_video(path: str) -> Tuple[float, int, int, bool, int]:
    """Return (fps, width, height, has_audio, n_frames) for a video file."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", path]
    try:
        raw  = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(raw)
    except Exception as exc:
        sys.exit(f"ffprobe failed on '{path}': {exc}")

    video_s   = None
    has_audio = False
    for s in data.get("streams", []):
        ct = s.get("codec_type")
        if ct == "video" and video_s is None:
            video_s = s
        elif ct == "audio":
            has_audio = True

    if video_s is None:
        sys.exit(f"No video stream found in: {path}")

    w, h = int(video_s["width"]), int(video_s["height"])
    num, den = video_s.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den)

    n_frames = int(video_s.get("nb_frames", 0))
    if n_frames <= 0:
        # nb_frames absent for some containers (MKV, etc.) — derive from duration
        duration = float(data.get("format", {}).get("duration", 0))
        n_frames = max(1, int(round(duration * fps)))

    return fps, w, h, has_audio, n_frames


# ============================================================
#  FFMPEG WRITER
# ============================================================

def spawn_writer(output_path: str, fps: float,
                 width: int, height: int,
                 audio_source: Optional[str] = None) -> subprocess.Popen:
    """
    Spawn ffmpeg reading raw BGR24 frames from stdin.
    If `audio_source` is given, its first audio stream is copied to the output.
    Output is H.264 CRF-16, yuv420p for broad compatibility.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_source:
        cmd += ["-i", audio_source]

    cmd += [
        "-vcodec", "libx264",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    if audio_source:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]

    cmd.append(output_path)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def finish_writer(proc: subprocess.Popen) -> None:
    """Close stdin, wait for ffmpeg to finish, and exit cleanly on error."""
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"[ffmpeg error]\n{stderr.decode()}", file=sys.stderr)
        sys.exit(1)


def _write_frame(proc: subprocess.Popen, frame: np.ndarray) -> None:
    """Write one raw frame to ffmpeg stdin; surface ffmpeg's error on pipe break."""
    try:
        proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        # ffmpeg exited early — grab its error output and abort
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode(errors="replace")
        sys.exit(f"[ffmpeg exited early]\n{err or '(no stderr output)'}")


# ============================================================
#  PER-INPUT-TYPE PROCESSORS
# ============================================================

def process_video(args: argparse.Namespace) -> None:
    fps, w, h, has_audio, n_frames = probe_video(args.input)
    # If cropping, use crop dimensions; otherwise use full video dimensions
    if args.crop:
        _, _, w, h = args.crop
    # yuv420p requires even dimensions; trim one pixel if needed
    w, h = w - w % 2, h - h % 2
    print(f"[video]   {w}×{h}  {fps:.3f} fps  {n_frames} frames  audio={has_audio}")

    tx, ty, theta, zoom_var = build_motion(n_frames, fps, args.intensity, args.seed,
                                            args.direction, args.speed)
    base_zoom = compute_zoom(tx, ty, theta, zoom_var, w, h)

    print(f"[motion]  zoom={base_zoom:.4f}×  "
          f"tx={np.max(np.abs(tx)):.2f}px  ty={np.max(np.abs(ty)):.2f}px  "
          f"θ={np.max(np.abs(theta)):.4f}°  zoom_var=±{np.max(np.abs(zoom_var))*100:.2f}%  "
          f"dir={args.direction}  speed={args.speed}")

    proc    = spawn_writer(args.output, fps, w, h,
                           audio_source=args.input if has_audio else None)
    cap     = cv2.VideoCapture(args.input)
    written = 0
    try:
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Apply user crop first, then trim to even dimensions
            if args.crop:
                cx, cy, cw, ch = args.crop
                frame = frame[cy:cy+ch, cx:cx+cw]
            frame = frame[:h, :w]
            warped = warp_frame(frame, float(tx[i]), float(ty[i]),
                                 float(theta[i]), base_zoom + float(zoom_var[i]), w, h)
            _write_frame(proc, warped)
            written += 1
            if (i + 1) % 30 == 0 or (i + 1) == n_frames:
                print(f"  frame {i+1}/{n_frames} "
                      f"({100*(i+1)/n_frames:.0f}%)", flush=True)
    finally:
        cap.release()

    finish_writer(proc)
    print(f"[done]    wrote {written} frames → {args.output}")


def process_image(args: argparse.Namespace) -> None:
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Cannot read image: {args.input}")

    # Apply crop before anything else
    if args.crop:
        cx, cy, cw, ch = args.crop
        img = img[cy:cy+ch, cx:cx+cw]

    h, w = img.shape[:2]
    # yuv420p requires even dimensions; crop one pixel if needed
    w, h = w - w % 2, h - h % 2
    img  = img[:h, :w]

    fps      = args.fps
    n_frames = max(1, int(round(args.duration * fps)))

    print(f"[image]   {w}×{h}  →  {n_frames} frames @ {fps} fps  ({args.duration:.1f} s)")

    tx, ty, theta, zoom_var = build_motion(n_frames, fps, args.intensity, args.seed,
                                            args.direction, args.speed)
    base_zoom = compute_zoom(tx, ty, theta, zoom_var, w, h)

    print(f"[motion]  zoom={base_zoom:.4f}×  "
          f"tx={np.max(np.abs(tx)):.2f}px  ty={np.max(np.abs(ty)):.2f}px  "
          f"θ={np.max(np.abs(theta)):.4f}°  zoom_var=±{np.max(np.abs(zoom_var))*100:.2f}%  "
          f"dir={args.direction}  speed={args.speed}")

    proc = spawn_writer(args.output, fps, w, h)

    for i in range(n_frames):
        warped = warp_frame(img, float(tx[i]), float(ty[i]),
                             float(theta[i]), base_zoom + float(zoom_var[i]), w, h)
        _write_frame(proc, warped)
        if (i + 1) % 30 == 0 or (i + 1) == n_frames:
            print(f"  frame {i+1}/{n_frames} "
                  f"({100*(i+1)/n_frames:.0f}%)", flush=True)

    finish_writer(proc)
    print(f"[done]    wrote {n_frames} frames → {args.output}")


# ============================================================
#  INPUT-TYPE DETECTION
# ============================================================

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts", ".ts"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def detect_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    # Extension ambiguous — probe with OpenCV
    cap = cv2.VideoCapture(path)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
        cap.release()
        return "video"
    cap.release()
    if cv2.imread(path) is not None:
        return "image"
    sys.exit(f"Cannot determine input type for: {path}")


# ============================================================
#  ENTRY POINT
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply subtle handheld camera motion to a video or image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",     required=True,
                    help="Input video file or image")
    ap.add_argument("--output",    required=True,
                    help="Output .mp4 file")
    ap.add_argument("--intensity", type=float, default=1.0,
                    help="Motion scale multiplier (1.0 = subtly handheld)")
    ap.add_argument("--seed",      type=int,   default=42,
                    help="RNG seed — same seed reproduces identical motion")
    ap.add_argument("--fps",       type=float, default=30.0,
                    help="Output frame rate (image input only; video uses source fps)")
    ap.add_argument("--duration",  type=float, default=6.0,
                    help="Clip length in seconds (image input only; video uses source duration)")
    ap.add_argument("--crop",      type=int,   nargs=4, metavar=("X","Y","W","H"),
                    default=None,
                    help="Crop source before processing: X Y W H in pixels")
    ap.add_argument("--direction", choices=["balanced","horizontal","vertical"],
                    default="balanced",
                    help="Axis bias for the shake motion (default: balanced)")
    ap.add_argument("--speed",     type=float, default=5.0,
                    help="Motion speed 1–10: 1=slow lazy drift, 5=default, 10=rapid erratic (default: 5)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}")

    kind = detect_kind(args.input)
    (process_video if kind == "video" else process_image)(args)


if __name__ == "__main__":
    main()
