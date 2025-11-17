# Chordify — Rebuilt with Grad-CAM XAI

This repo is a rebuilt end-to-end chord recognition pipeline with Grad-CAM explainability.
Data: McGill–Billboard dataset (use your local copy in `data/`). See: https://www.kaggle.com/datasets/jacobvs/mcgill-billboard. :contentReference[oaicite:5]{index=5}

Quickstart:
1. Install dependencies:
   pip install -r requirements.txt

2. Prepare data:
   Place audio files and annotation `.lab` files under `data/`. Update `notebooks/xai_demo.ipynb` with paths.

3. (Optional) Train a tiny model for demo:
   python -c "from src.training.train import simple_train; ..."

4. Run Grad-CAM demo:
   python src/inference/predict_and_xai.py --audio data/example.wav --ckpt checkpoints/model.pt --out outputs/xai_examples/example_overlay.png

Notes:
- The annotation parser expects simple `.lab` interval files (start end label). If your annotations are in JAMS or other formats, adapt `src/annotations/parser.py`.
- Grad-CAM attaches to the last Conv2D in the feature extractor by default. Adjust if your model differs.
