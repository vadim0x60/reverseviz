# Reverseviz (reverse music visualizer)

In Soviet Russia, music dances to you.

`reverseviz.py` turns **webcam motion** into **generative music**: instead of visuals reacting to audio, audio reacts to motion.

## What it does

- Captures consecutive webcam frames.
- Computes a simple motion signal from the per-row absolute frame difference.
- Extracts stable features:
  - `E` (intensity): overall motion energy, `0..1`
  - `dE` (onset): positive change in intensity
  - `C` (centroid): where motion is happening vertically, `0..1` (top → bottom)
- Maps those features to sound.

## Requirements

- Python (tested on 3.x)
- A working webcam
- Audio output device

Python deps:

```bash
pip install opencv-python numpy pyo
```

### macOS notes (pyo)

`pyo` may require PortAudio and build tools.

```bash
brew install portaudio
```

If `pyo` prints warnings about missing GUI toolkits (WxPython/Tkinter), that’s fine unless you’re trying to use `pyo`’s GUI features.

## Quickstart

Run the default mode (EDM-ish sequencer):

```bash
python3 reverseviz.py
```

Show a debug video window (press `q` to quit):

```bash
python3 reverseviz.py --show
```

**Live Tuning (DJ Mode):**

Open an interactive window with sliders to tweak motion sensitivity, mix levels, and synth parameters in real-time:

```bash
python3 reverseviz.py --dj
```

List audio output devices and pick one:

```bash
python3 reverseviz.py --list-audio-devices
python3 reverseviz.py --output-device 3
```

## Modes

### `edm` (default)

A fixed-tempo 16-step sequencer with a stable scale (A minor pentatonic). Motion controls groove intensity, accents, note choice, and timbre.

- **Spectral Filter Bank**: In this mode, your vertical pose (the motion vector) directly shapes the timbre of the lead synth via a 48-band filter bank. Raising your hands or moving in specific vertical zones carves out different frequencies.

Useful options:

- `--bpm 120`
- `--master-amp 0.8`
- `--min-intensity 0.01` (below this, it stays in a lighter “idle groove”)
- `--idle-level 0.08` (how alive it feels when still)
- `--hype-intensity 0.55` / `--hype-onset 0.35` (big-move accents)

### `pitch`

Legacy mode that maps motion centroid to continuous pitch (smooth glide).

Useful options:

- `--f-low 110 --f-high 1760`
- `--pitch-amp 0.35`
- `--pitch-glide 0.05`

Run it:

```bash
python3 reverseviz.py --mode pitch
```

## Tuning the motion detector

If it feels too sensitive or too dead, these usually matter most (all can be tuned live using `--dj`):

- `--diff-threshold` (ignore tiny pixel changes)
- `--blur` (smooth noise; must be odd internally)
- `--bins` (vertical resolution of motion features)
- `--intensity-gain` / `--intensity-ema` / `--onset-gain` (feature smoothing + scaling)

Example: more stable / less twitchy:

```bash
python3 reverseviz.py --diff-threshold 18 --blur 7 --intensity-ema 0.9
```

## Troubleshooting

- **Camera won’t open / black frames**: check OS camera permissions and try `--camera 1` (or another index).
- **No audio**: verify your system output device, then try `--list-audio-devices` and pass `--output-device`.
- **`Missing dependency: pyo`**: `pip install pyo` may need system audio libs (see macOS notes).
- **Debug window doesn’t appear**: use `--show`; quit with `q` in the window.

## Repo layout

- `reverseviz.py` — main script (motion extraction + audio synthesis)
- `README.md` — this file
