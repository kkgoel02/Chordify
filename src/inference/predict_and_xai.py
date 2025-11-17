# src/inference/predict_and_xai.py
import os
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa

from src.models.cnn_chroma import SmallChromaCNN
from src.xai.gradcam_chroma import GradCAMChroma

DEFAULT_WINDOW = 64

def load_label_map(path="label_map.json"):
    with open(path, "r", encoding="utf-8") as fh:
        lm = json.load(fh)
    idx2label = {v: k for k, v in lm.items()}
    return lm, idx2label

def chroma_from_audio(audio_path, sr=22050, hop_length=512, n_fft=2048):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma.astype(np.float32)  # (12, T)

def save_overlay(chroma, heatmap, out_path, alpha=0.6, figsize=(8,4)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.axis("off")
    ax.imshow(chroma, origin="lower", aspect="auto")
    ax.imshow(heatmap, origin="lower", aspect="auto", cmap="jet", alpha=alpha)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def predict_on_chroma_matrix(chroma, ckpt, label_map_path="label_map.json", window_frames=DEFAULT_WINDOW, out_dir=None, do_xai=True):
    """
    chroma: np.array (12, T)
    """
    lm, idx2label = load_label_map(label_map_path)
    n_classes = len(lm)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SmallChromaCNN(n_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # find last Conv2d in conv_block
    target_layer = None
    for m in reversed(list(model.conv_block)):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break
    if target_layer is None:
        target_layer = model.conv_block[-1]

    gcam = GradCAMChroma(model, target_layer) if do_xai else None

    T = chroma.shape[1]
    half = window_frames // 2
    preds = []
    confs = []
    heatmaps = []

    for center in range(half, T - half):
        patch = chroma[:, center-half:center+half]
        if patch.shape[1] < window_frames:
            patch = np.pad(patch, ((0,0),(0, window_frames - patch.shape[1])), mode='constant')
        x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,12,W)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            cls = int(prob.argmax(dim=1).item())
            conf = float(prob[0, cls].cpu().item())
            preds.append(idx2label[cls])
            confs.append(conf)
        if do_xai:
            hm, pcls, conf2 = gcam(x)
            heatmaps.append(hm.copy())

    # save central overlay and preds if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        center = T//2
        patch = chroma[:, center-half:center+half]
        if patch.shape[1] < window_frames:
            patch = np.pad(patch, ((0,0),(0, window_frames - patch.shape[1])), mode='constant')
        if do_xai and len(heatmaps) > 0:
            hm = heatmaps[len(heatmaps)//2]
            save_overlay(patch, hm, os.path.join(out_dir, "xai_overlay.png"))
        with open(os.path.join(out_dir, "preds.txt"), "w", encoding="utf-8") as fh:
            for p, c in zip(preds, confs):
                fh.write(f"{p}\t{c:.4f}\n")

    return preds, confs, heatmaps

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--track", help="Track id (to use CSV chroma from metadata)", default=None)
    p.add_argument("--audio", help="Path to audio file (wav). If provided, will compute chroma from audio.", default=None)
    p.add_argument("--ckpt", required=True, help="Checkpoint path (chroma model)")
    p.add_argument("--out", help="Output folder for overlay + preds", default="outputs/xai_examples")
    args = p.parse_args()

    if args.audio:
        chroma = chroma_from_audio(args.audio)
    elif args.track:
        csvp = os.path.join("data", "raw", "metadata", args.track, "bothchroma.csv")
        chroma = np.loadtxt(csvp, delimiter=",").T
    else:
        raise RuntimeError("Either --audio or --track <id> must be provided")

    preds, confs, heatmaps = predict_on_chroma_matrix(chroma, args.ckpt, out_dir=args.out)
    print(f"[INFO] Saved demo outputs to {args.out}")
