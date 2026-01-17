# Reverse music visualizer

In Soviet Russia, music dances to you.

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
