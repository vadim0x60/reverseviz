#!/usr/bin/env python3
"""Webcam motion -> EDM-ish music (cv2 + pyo).

This is a musical evolution of the original "motion -> continuous pitch" idea.

Approach
- Grab consecutive webcam frames.
- Compute a per-row motion vector from |frame_t - frame_{t-1}| summed horizontally.
- Reduce the vector into vertical bands.
- Extract a few stable motion features:
  - E: intensity (0..1)
  - dE: onset (positive intensity change)
  - C: centroid (0..1, top->bottom)
- Run a fixed-tempo (default: 100 BPM) 16-step sequencer.
- Use E/dE/C to control drum hits, note choice, and timbre.

Modes
- edm  : fixed BPM, stable harmony (A minor pentatonic), reactive groove.
- pitch: legacy continuous pitch mapping.

Dependencies
  pip install opencv-python numpy pyo

On macOS, `pyo` may require PortAudio and other build deps.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: opencv-python (cv2)") from e

try:
    from pyo import (
        Adsr,
        ButBP,
        ButHP,
        ButLP,
        Compress,
        Delay,
        Disto,
        Freeverb,
        Metro,
        Mix,
        Noise,
        Osc,
        SawTable,
        Server,
        Sig,
        SigTo,
        Sine,
        TrigFunc,
    )
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pyo") from e


@dataclass
class MotionToMusicConfig:
    mode: str = "edm"  # edm|pitch

    # Camera
    camera: int = 0
    width: int = 320
    height: int = 240
    fps: float = 30.0

    # Motion processing
    bins: int = 48
    blur: int = 5
    diff_threshold: int = 12
    vec_ema: float = 0.6

    # Feature mapping
    intensity_gain: float = 18.0
    intensity_ema: float = 0.85
    onset_gain: float = 2.5

    # When intensity is below this, we still keep a quiet "idle" groove,
    # but we stop the heavy stuff (kicks/fills/claps, big lead hits).
    min_intensity: float = 0.01

    # Idle groove level (0..1). Keeps the system obviously "alive".
    idle_level: float = 0.08

    # EDM
    bpm: float = 100.0
    master_amp: float = 0.9

    # "Bombastic move" detection.
    # If either (E > hype_intensity) or (dE > hype_onset), we trigger a one-shot
    # accent (crash / impact) and temporarily hype the lead.
    hype_intensity: float = 0.8
    hype_onset: float = 0.15
    hype_cooldown_steps: int = 8

    # Mix levels (pre-master)
    kick_level: float = 1.0
    clap_level: float = 0.7
    hat_level: float = 0.35
    bass_level: float = 0.65
    lead_level: float = 0.55

    # Legacy pitch mode
    f_low: float = 110.0
    f_high: float = 1760.0
    pitch_amp: float = 0.35
    pitch_glide_sec: float = 0.05

    # Audio
    output_device: int | None = None

    # UI
    show: bool = False
    debug: bool = False


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def midi_to_hz(m: float) -> float:
    return float(440.0 * (2.0 ** ((m - 69.0) / 12.0)))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def log_interp(f_low: float, f_high: float, t01: float) -> float:
    t01 = float(np.clip(t01, 0.0, 1.0))
    return float(np.exp(np.log(f_low) + (np.log(f_high) - np.log(f_low)) * t01))


def bin_rows(row_energy: np.ndarray, bins: int) -> np.ndarray:
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


def weighted_centroid(vec_top_to_bottom: np.ndarray) -> float:
    vec = np.asarray(vec_top_to_bottom, dtype=np.float32)
    total = float(vec.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return 0.5
    idx = np.arange(vec.shape[0], dtype=np.float32)
    center = float((vec * idx).sum() / total)  # 0..bins-1 (top..bottom)
    return float(center / max(1.0, float(vec.shape[0] - 1)))  # 0..1 top..bottom


def intensity_from_vec(vec: np.ndarray, gain: float) -> float:
    # vec values are already ~0..1-ish; average then gain.
    return clip01(float(vec.mean()) * float(gain))


def quantize_scale_degree(t01: float, degrees: list[int], root_midi: int, octaves: int = 2) -> float:
    """Map 0..1 to a note in a repeating scale."""
    t01 = float(np.clip(t01, 0.0, 1.0))
    steps = degrees * max(1, int(octaves))
    octave_offsets = []
    for o in range(max(1, int(octaves))):
        octave_offsets.extend([d + 12 * o for d in degrees])
    idx = int(round(t01 * (len(octave_offsets) - 1)))
    return float(root_midi + octave_offsets[idx])


def parse_args() -> MotionToMusicConfig:
    p = argparse.ArgumentParser(description="Webcam motion -> EDM-ish music")

    p.add_argument("--mode", choices=["edm", "pitch"], default="edm")

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=float, default=30.0)

    p.add_argument("--bins", type=int, default=48)
    p.add_argument("--blur", type=int, default=5)
    p.add_argument("--diff-threshold", type=int, default=12)
    p.add_argument("--vec-ema", type=float, default=0.6)

    p.add_argument("--intensity-gain", type=float, default=18.0)
    p.add_argument("--intensity-ema", type=float, default=0.85)
    p.add_argument("--onset-gain", type=float, default=2.5)
    p.add_argument("--min-intensity", type=float, default=0.01)
    p.add_argument(
        "--idle-level",
        type=float,
        default=0.08,
        help="Idle groove amplitude (0..1) when there is no motion",
    )

    p.add_argument("--bpm", type=float, default=100.0)
    p.add_argument("--master-amp", type=float, default=0.9)

    p.add_argument("--hype-intensity", type=float, default=0.55)
    p.add_argument("--hype-onset", type=float, default=0.35)
    p.add_argument("--hype-cooldown-steps", type=int, default=8)

    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--list-audio-devices", action="store_true")

    # Legacy pitch options
    p.add_argument("--f-low", type=float, default=110.0)
    p.add_argument("--f-high", type=float, default=1760.0)
    p.add_argument("--pitch-amp", type=float, default=0.35)
    p.add_argument("--pitch-glide", type=float, default=0.05)

    p.add_argument("--show", action="store_true")
    p.add_argument("--debug", action="store_true")

    a = p.parse_args()

    if a.list_audio_devices:
        from pyo import pa_get_default_output, pa_get_output_devices

        names, idxs = pa_get_output_devices()
        print("Default output:", pa_get_default_output())
        for name, idx in zip(names, idxs):
            print(f"{idx}: {name}")
        raise SystemExit(0)

    return MotionToMusicConfig(
        mode=a.mode,
        camera=a.camera,
        width=a.width,
        height=a.height,
        fps=a.fps,
        bins=a.bins,
        blur=a.blur,
        diff_threshold=a.diff_threshold,
        vec_ema=a.vec_ema,
        intensity_gain=a.intensity_gain,
        intensity_ema=a.intensity_ema,
        onset_gain=a.onset_gain,
        min_intensity=a.min_intensity,
        idle_level=a.idle_level,
        bpm=a.bpm,
        master_amp=a.master_amp,
        hype_intensity=a.hype_intensity,
        hype_onset=a.hype_onset,
        hype_cooldown_steps=a.hype_cooldown_steps,
        output_device=a.output_device,
        f_low=a.f_low,
        f_high=a.f_high,
        pitch_amp=a.pitch_amp,
        pitch_glide_sec=a.pitch_glide,
        show=a.show,
        debug=a.debug,
    )


def build_pitch_mode(cfg: MotionToMusicConfig):
    freq_sig = SigTo(value=cfg.f_low, time=cfg.pitch_glide_sec, init=cfg.f_low)
    amp_sig = SigTo(value=0.0, time=cfg.pitch_glide_sec, init=0.0)
    osc = Sine(freq=freq_sig, mul=amp_sig * cfg.pitch_amp)
    comp = Compress(osc, thresh=-20, ratio=6, risetime=0.01, falltime=0.1)
    out = comp.out()

    # Keep references to avoid GC-induced crashes in some pyo builds.
    keep = {
        "freq_sig": freq_sig,
        "amp_sig": amp_sig,
        "osc": osc,
        "comp": comp,
        "out": out,
    }
    return freq_sig, amp_sig, keep


def build_edm_mode(cfg: MotionToMusicConfig, motion_state: dict):
    """Create instruments and a step callback.

    motion_state is a dict updated by the video loop:
      E (0..1), dE (0..1), C (0..1 top..bottom)
    """

    # --- Instruments ---
    # Kick: pitch drop sine + click.
    kick_amp = Sig(0.0)
    kick_env = Adsr(attack=0.001, decay=0.10, sustain=0.0, release=0.0, dur=0.12, mul=kick_amp)
    kick_freq = SigTo(value=55.0, time=0.02, init=55.0)
    kick = Sine(freq=kick_freq, mul=kick_env)

    kick_click_amp = Sig(0.0)
    kick_click_env = Adsr(attack=0.001, decay=0.02, sustain=0.0, release=0.0, dur=0.03, mul=kick_click_amp)
    kick_click = ButHP(Noise(mul=kick_click_env), freq=3000)

    # Clap: noise burst bandpassed.
    clap_amp = Sig(0.0)
    clap_env = Adsr(attack=0.001, decay=0.12, sustain=0.0, release=0.0, dur=0.14, mul=clap_amp)
    clap = ButBP(Noise(mul=clap_env), freq=2000, q=6)

        # Hats: bright noise tick.
    hat_amp = Sig(0.0)
    hat_env = Adsr(attack=0.001, decay=0.05, sustain=0.0, release=0.0, dur=0.06, mul=hat_amp)
    hat = ButHP(Noise(mul=hat_env), freq=6500)

    # Crash/impact: bright noise burst into reverb (triggered on big moves).
    crash_amp = Sig(0.0)
    crash_env = Adsr(attack=0.001, decay=0.9, sustain=0.0, release=0.0, dur=0.95, mul=crash_amp)
    crash_src = ButHP(Noise(mul=crash_env), freq=2500)
    crash = Freeverb(crash_src, size=0.93, damp=0.5, bal=0.55)


    # Bass: sine + a touch of harmonics, lowpassed.
    bass_amp = Sig(0.0)
    bass_env = Adsr(attack=0.002, decay=0.18, sustain=0.0, release=0.0, dur=0.22, mul=bass_amp)
    bass_freq = SigTo(value=55.0, time=0.02, init=55.0)
    bass_src = Sine(freq=bass_freq, mul=bass_env) + (Sine(freq=bass_freq * 2.0, mul=bass_env * 0.18))
    bass_cut = SigTo(value=120.0, time=0.05, init=120.0)
    bass = ButLP(bass_src, freq=bass_cut)

    # Lead: saw oscillator, filtered, delayed + reverb.
    lead_amp = Sig(0.0)
    lead_env = Adsr(attack=0.003, decay=0.12, sustain=0.0, release=0.0, dur=0.16, mul=lead_amp)
    lead_freq = SigTo(value=440.0, time=0.03, init=440.0)
    lead_table = SawTable(order=12)
    lead_osc = Osc(table=lead_table, freq=lead_freq, mul=lead_env)
    lead_cut = SigTo(value=800.0, time=0.06, init=800.0)
    lead_filt = ButLP(lead_osc, freq=lead_cut)
    lead_sat = Disto(lead_filt, drive=0.75, slope=0.85, mul=1.0)
    lead_del = Delay(lead_sat, delay=0.30, feedback=0.25, mul=0.55)  # ~8th note at 100 BPM
    lead_rev = Freeverb(lead_sat + lead_del, size=0.82, damp=0.6, bal=0.22)

    # Mix
    drums = (kick + kick_click) * cfg.kick_level + clap * cfg.clap_level + hat * cfg.hat_level
    fx = crash * 0.9
    melodic = bass * cfg.bass_level + (lead_sat + lead_rev) * cfg.lead_level
    pre = Mix(drums + melodic + fx, voices=2)

    master = Compress(pre, thresh=-18, ratio=5, risetime=0.01, falltime=0.12)
    out = master.out()

    # --- Sequencer ---
    bpm = float(cfg.bpm)
    step_time = 60.0 / bpm / 4.0  # 16th notes
    metro = Metro(time=step_time).play()

    # A minor pentatonic (degrees relative to A): A C D E G
    degrees = [0, 3, 5, 7, 10]
    root_midi = 57  # A3

    step = {"i": -1}
    hype = {"cooldown": 0}

    def on_step():
        step["i"] = (step["i"] + 1) % 16
        i = step["i"]

        E = float(motion_state.get("E", 0.0))
        dE = float(motion_state.get("dE", 0.0))
        C = float(motion_state.get("C", 0.5))

        # When still, keep a quiet "idle" groove so the connection is
        # obvious: motion makes it explode; stillness makes it die down.
        active = E >= cfg.min_intensity
        idle = float(cfg.idle_level)

        # --- Hype detection ---
        if hype["cooldown"] > 0:
            hype["cooldown"] -= 1

        is_hype = (E >= float(cfg.hype_intensity)) or (dE >= float(cfg.hype_onset))
        if active and is_hype and hype["cooldown"] == 0:
            # Big gesture => obvious impact.
            crash_amp.value = clip01(0.35 + 0.95 * dE + 0.55 * E)
            crash_env.play()
            hype["cooldown"] = int(cfg.hype_cooldown_steps)

        # --- Kick ---
        # Keep a quiet kick on the grid when idle; go full power when active.
        is_four_on_floor = i in (0, 4, 8, 12)
        fill_prob = clip01(0.05 + 0.35 * dE)
        do_fill = (not is_four_on_floor) and active and (random.random() < fill_prob)
        if is_four_on_floor or do_fill:
            vel = (0.25 + 0.50 * E + 0.85 * dE) if active else (0.20 * idle)
            vel = clip01(vel)
            kick_amp.value = vel
            kick_click_amp.value = vel * 0.6
            # quick pitch drop
            kick_freq.value = 110.0
            kick_freq.time = 0.03
            kick_freq.value = 48.0
            kick_env.play()
            kick_click_env.play()

        # --- Clap ---
        if active and i in (4, 12):
            vel = clip01(0.18 + 1.10 * dE + 0.45 * E)
            clap_amp.value = vel
            clap_env.play()

        # --- Hats (8th-note drive) ---
        # Idle hats keep the groove alive.
        if i % 2 == 0:
            vel = (0.06 + 0.32 * E + 0.25 * dE) if active else (0.12 * idle)
            hat_amp.value = clip01(vel)
            hat_env.play()

        # --- Bass ---
        # Idle bass is a soft root pulse.
        if i in (0, 8):
            vel = (0.16 + 0.55 * E) if active else (0.20 * idle)
            bass_amp.value = clip01(vel)
            bass_note = 45  # A2
            if active and C > 0.62 and (random.random() < clip01(0.25 + 0.35 * E)):
                bass_note = 52  # E3
            bass_freq.value = midi_to_hz(bass_note)
            bass_cut.value = 90.0 + ((E if active else idle) * 1600.0)
            bass_env.play()

        # --- Lead ---
        # When idle: sparse, quiet lead to keep the "dancer controls it" link.
        if i in (2, 6, 10, 14):
            vel = (0.10 + 0.45 * E + 0.55 * dE) if active else (0.10 * idle)
            vel = clip01(vel)

            # In hype moments, make it *obviously* pop:
            # - higher velocity
            # - octave jump when dancer is "up" (centroid near top)
            # - brighter filter
            hype_boost = 1.0
            octave_boost = 0
            if active and is_hype:
                hype_boost = 1.0 + 0.9 * clip01(max(E - cfg.hype_intensity, dE - cfg.hype_onset) + 0.35)
                if C < 0.45:
                    octave_boost = 12

            if active or (random.random() < 0.35):
                lead_amp.value = clip01(vel * hype_boost)
                # Top is "high" (invert centroid).
                t = 1.0 - float(np.clip(C, 0.0, 1.0))
                midi = quantize_scale_degree(t, degrees=degrees, root_midi=root_midi, octaves=2) + octave_boost
                lead_freq.value = midi_to_hz(midi)
                lead_cut.value = 500.0 + ((E if active else idle) * 5000.0) + (1500.0 if (active and is_hype) else 0.0)
                lead_env.play()

        if cfg.debug and i == 0:
            print(f"E={E:0.3f} dE={dE:0.3f} C={C:0.3f}")

    trig = TrigFunc(metro, on_step)

    # Return objects to keep references alive.
    return {
        "out": out,
        "master": master,
        "pre": pre,
        "drums": drums,
        "melodic": melodic,
        "kick": kick,
        "kick_env": kick_env,
        "kick_freq": kick_freq,
        "kick_click": kick_click,
        "kick_click_env": kick_click_env,
        "clap": clap,
        "clap_env": clap_env,
        "hat": hat,
        "hat_env": hat_env,
        "crash": crash,
        "crash_env": crash_env,
        "bass": bass,
        "bass_env": bass_env,
        "bass_freq": bass_freq,
        "lead_sat": lead_sat,
        "lead_env": lead_env,
        "lead_freq": lead_freq,
        "lead_del": lead_del,
        "lead_rev": lead_rev,
        "metro": metro,
        "trig": trig,
        "step_time": step_time,
    }


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
    s = Server(duplex=0)
    if cfg.output_device is not None:
        s.setOutputDevice(int(cfg.output_device))
    s.setAmp(float(cfg.master_amp))
    s.deactivateMidi()
    s.boot()
    s.start()

    # Shared motion state (used by the EDM sequencer callback).
    motion_state: dict[str, float] = {"E": 0.0, "dE": 0.0, "C": 0.5}

    # Build audio graph.
    pitch_freq = None
    pitch_amp = None
    audio_objs = {}
    if cfg.mode == "pitch":
        pitch_freq, pitch_amp, audio_objs = build_pitch_mode(cfg)
    else:
        audio_objs = build_edm_mode(cfg, motion_state)

    prev_gray: np.ndarray | None = None
    smoothed_vec: np.ndarray | None = None

    E_ema = 0.0

    target_period = 1.0 / max(1.0, cfg.fps)

    print(f"Running mode={cfg.mode}. Press Ctrl+C to stop.")
    if cfg.mode == "edm":
        print(f"EDM clock: {cfg.bpm:.1f} BPM (8th hats, stable A minor).")
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

            # Per-row motion energy: sum horizontally, normalize to ~0..1
            row_energy = diff.sum(axis=1).astype(np.float32)
            row_energy /= float(diff.shape[1] * 255)

            vec = bin_rows(row_energy, cfg.bins)

            if smoothed_vec is None:
                smoothed_vec = vec
            else:
                smoothed_vec = (float(cfg.vec_ema) * smoothed_vec) + ((1.0 - float(cfg.vec_ema)) * vec)

            C = weighted_centroid(smoothed_vec)  # 0..1 top..bottom
            E = intensity_from_vec(smoothed_vec, cfg.intensity_gain)

            E_ema = (float(cfg.intensity_ema) * E_ema) + ((1.0 - float(cfg.intensity_ema)) * E)
            dE = clip01((E - E_ema) * float(cfg.onset_gain))

            motion_state["C"] = C
            motion_state["E"] = E
            motion_state["dE"] = dE

            if cfg.mode == "pitch" and pitch_freq is not None and pitch_amp is not None:
                # Map centroid to pitch (top=high).
                t = 1.0 - float(np.clip(C, 0.0, 1.0))
                freq = log_interp(cfg.f_low, cfg.f_high, t)
                amp = E
                if amp < cfg.min_intensity:
                    amp = 0.0
                pitch_freq.value = freq
                pitch_amp.value = amp

            if cfg.show:
                h, w = frame.shape[:2]
                vis = frame.copy()
                y = int(float(np.clip(C, 0.0, 1.0)) * (h - 1))
                cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"E {E:.2f} dE {dE:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("motion", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            dt = time.time() - t0
            if dt < target_period:
                time.sleep(target_period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop audio callbacks/objects before shutting down the server.
        try:
            metro = audio_objs.get("metro")
            if metro is not None:
                metro.stop()
        except Exception:
            pass
        try:
            trig = audio_objs.get("trig")
            if trig is not None:
                trig.stop()
        except Exception:
            pass
        try:
            out = audio_objs.get("out")
            if out is not None:
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
