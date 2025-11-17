# Chordify — Chroma-based Chord Recognition + Grad-CAM XAI

Project scaffold for training a chroma-based CNN on the McGill Billboard style CSV chroma + .lab annotations,
and for running inference on CSV chroma or audio (librosa chroma extraction) with Grad-CAM explainability.

## Quick steps

1. Place dataset as:
   - `data/raw/annotations/<trackID>/*.lab`
   - `data/raw/metadata/<trackID>/bothchroma.csv` (frames x 12)
   - `data/raw/metadata/<trackID>/tuning.csv` (optional)

2. Build label map:
   ```bash
   python scripts/build_label_map.py
