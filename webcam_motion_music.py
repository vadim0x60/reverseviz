#!/usr/bin/env python3
"""Webcam motion to pitch (cv2 + pyo).

Idea
- Grab consecutive webcam frames.
- Compute per-row motion energy from |frame_t - frame_{t-1}| summed horizontally.
- Map row index -> pitch (top = high, bottom = low).
- Convert the motion vector into a single dominant pitch via a weighted average,
  and loudness via total motion energy.

Notes
- Using *every* row as an independent oscillator is usually too CPU-heavy.
  This script bins rows into a smaller number of bands (default: 48) while
  still following the “pitch vector” concept.

Usage
  python webcam_motion_music.py
  python webcam_motion_music.py --camera 0 --bins 64 --min-motion 0.015

Dependencies
  pip install opencv-python numpy pyo

On macOS, `pyo` may require PortAudio and other build deps.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: opencv-python (cv2)") from e

try:
    from pyo import Compress, Server, Sine, SigTo
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pyo") from e


@dataclass
class MotionToPitchConfig:
    camera: int = 0
    width: int = 320
    height: int = 240
    fps: float = 30.0
    bins: int = 48

    # Audio
    output_device: int | None = None

    # Motion processing
    blur: int = 5
    diff_threshold: int = 12
    ema: float = 0.6
    motion_gain: float = 8.0
    min_motion: float = 0.005

    # Sound mapping
    f_low: float = 110.0
    f_high: float = 1760.0
    amp: float = 0.35
    glide_sec: float = 0.05

    # UI / debug
    show: bool = False
    debug: bool = False


def parse_args() -> MotionToPitchConfig:
    p = argparse.ArgumentParser(description="Play pitch based on per-row webcam motion.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--bins", type=int, default=48, help="Number of vertical pitch bands")

    p.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="PortAudio output device index (see --list-audio-devices)",
    )
    p.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print PortAudio devices and exit",
    )

    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd)")
    p.add_argument("--diff-threshold", type=int, default=12, help="Per-pixel diff threshold (0-255)")
    p.add_argument("--ema", type=float, default=0.6, help="EMA smoothing for motion vector (0-1)")
    p.add_argument(
        "--motion-gain",
        type=float,
        default=8.0,
        help="Gain applied to motion->amplitude mapping",
    )
    p.add_argument("--min-motion", type=float, default=0.005, help="Gate threshold for motion energy")

    p.add_argument("--f-low", type=float, default=110.0)
    p.add_argument("--f-high", type=float, default=1760.0)
    p.add_argument("--amp", type=float, default=0.35)
    p.add_argument("--glide", type=float, default=0.05)

    p.add_argument("--show", action="store_true", help="Show debug video window")
    p.add_argument("--debug", action="store_true", help="Print motion / pitch diagnostics")

    a = p.parse_args()

    if a.list_audio_devices:
        # Import here to avoid pyo import side-effects unless requested.
        from pyo import pa_get_default_output, pa_get_output_devices

        names, idxs = pa_get_output_devices()
        print("Default output:", pa_get_default_output())
        for name, idx in zip(names, idxs):
            print(f"{idx}: {name}")
        raise SystemExit(0)

    return MotionToPitchConfig(
        camera=a.camera,
        width=a.width,
        height=a.height,
        fps=a.fps,
        bins=a.bins,
        output_device=a.output_device,
        blur=a.blur,
        diff_threshold=a.diff_threshold,
        ema=a.ema,
        motion_gain=a.motion_gain,
        min_motion=a.min_motion,
        f_low=a.f_low,
        f_high=a.f_high,
        amp=a.amp,
        glide_sec=a.glide,
        show=a.show,
        debug=a.debug,
    )


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def log_interp(f_low: float, f_high: float, t01: float) -> float:
    """Interpolate frequencies logarithmically."""
    t01 = float(np.clip(t01, 0.0, 1.0))
    return float(np.exp(np.log(f_low) + (np.log(f_high) - np.log(f_low)) * t01))


def bin_rows(row_energy: np.ndarray, bins: int) -> np.ndarray:
    """Downsample per-row energy into `bins` bands (top->bottom)."""
    rows = row_energy.shape[0]
    if bins <= 0:
        raise ValueError("bins must be positive")
    if bins >= rows:
        return row_energy.astype(np.float32, copy=False)

    edges = np.linspace(0, rows, bins + 1, dtype=np.int32)
    out = np.zeros((bins,), dtype=np.float32)
    for i in range(bins):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = min(rows, a + 1)
        out[i] = float(row_energy[a:b].mean())
    return out


def weighted_pitch_from_vector(vec_top_to_bottom: np.ndarray, cfg: MotionToPitchConfig) -> tuple[float, float]:
    """Return (freq_hz, amp_0_1) derived from motion vector.

    - vec indexes go top->bottom.
    - pitch uses a weighted average: more motion at top => higher pitch.
    - amp uses the average per-bin energy times `motion_gain`.
    """
    vec = np.asarray(vec_top_to_bottom, dtype=np.float32)
    total = float(vec.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return cfg.f_low, 0.0

    idx = np.arange(vec.shape[0], dtype=np.float32)
    center = float((vec * idx).sum() / total)  # 0..bins-1 (top..bottom)

    # Map top (0) -> 1.0, bottom -> 0.0
    t = 1.0 - (center / max(1.0, float(vec.shape[0] - 1)))
    freq = log_interp(cfg.f_low, cfg.f_high, t)

    # Average energy per band is usually small; apply gain.
    amp = (total / float(vec.shape[0])) * float(cfg.motion_gain)
    amp = float(np.clip(amp, 0.0, 1.0))
    return freq, amp


def main() -> None:
    cfg = parse_args()

    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cfg.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    blur_k = _ensure_odd(max(1, cfg.blur))

    # Audio
    # - `duplex=0` avoids opening an input stream (prevents warnings when the
    #   default input device is mono, like many headset mics).
    # - `deactivateMidi()` avoids Portmidi warnings if you have no MIDI devices.
    s = Server(duplex=0)
    if cfg.output_device is not None:
        s.setOutputDevice(int(cfg.output_device))
    s.setAmp(1.0)
    s.deactivateMidi()
    s.boot()
    s.start()

    freq_sig = SigTo(value=cfg.f_low, time=cfg.glide_sec, init=cfg.f_low)
    amp_sig = SigTo(value=0.0, time=cfg.glide_sec, init=0.0)

    # Light compression helps get it audible without huge peaks.
    osc = Sine(freq=freq_sig, mul=amp_sig * cfg.amp)
    out = Compress(osc, thresh=-20, ratio=6, risetime=0.01, falltime=0.1).out()

    prev_gray: np.ndarray | None = None
    smoothed_vec: np.ndarray | None = None

    target_period = 1.0 / max(1.0, cfg.fps)

    print(
        "Running. Press Ctrl+C to stop. "
        "(Tip: wave hand near top for high notes.)"
    )
    if cfg.show:
        print("Showing video window. Press 'q' to quit.")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if blur_k > 1:
                gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            diff = cv2.absdiff(gray, prev_gray)
            prev_gray = gray

            if cfg.diff_threshold > 0:
                _, diff = cv2.threshold(diff, cfg.diff_threshold, 255, cv2.THRESH_TOZERO)

            # Per-row motion energy: sum horizontally, normalize to 0..1-ish
            row_energy = diff.sum(axis=1).astype(np.float32)
            row_energy /= float(diff.shape[1] * 255)

            vec = bin_rows(row_energy, cfg.bins)

            if smoothed_vec is None:
                smoothed_vec = vec
            else:
                smoothed_vec = (cfg.ema * smoothed_vec) + ((1.0 - cfg.ema) * vec)

            freq, amp = weighted_pitch_from_vector(smoothed_vec, cfg)

            if amp < cfg.min_motion:
                amp = 0.0

            freq_sig.value = freq
            amp_sig.value = amp

            if cfg.debug:
                print(f"freq={freq:7.1f}Hz amp={amp:0.3f} motion_avg={float(smoothed_vec.mean()):0.4f}")

            if cfg.show:
                # Draw a simple indicator for where the pitch is coming from.
                h, w = frame.shape[:2]
                vis = frame.copy()
                # Compute pitch row from current freq for visualization.
                t = (np.log(freq) - np.log(cfg.f_low)) / (np.log(cfg.f_high) - np.log(cfg.f_low))
                y = int((1.0 - float(np.clip(t, 0.0, 1.0))) * (h - 1))
                cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{freq:.0f} Hz amp {amp:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("motion->pitch", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            # Keep loop close to requested fps
            dt = time.time() - t0
            if dt < target_period:
                time.sleep(target_period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            out.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            s.stop()
            s.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
