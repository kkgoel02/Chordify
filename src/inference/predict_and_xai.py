# src/inference/predict_and_xai.py
import torch
import numpy as np
import os
from src.audio.features import load_audio, compute_mel_spectrogram, normalize_spectrogram
from src.models.cnn import SmallSpecCNN
from src.xai.gradcam import GradCAM, overlay_and_save

def find_last_conv(model):
    # find last Conv2d module inside model.features
    for m in reversed(list(model.features)):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

def run_xai_for_audio(audio_path, ckpt_path, out_png, n_classes=25, window_frames=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y, sr = load_audio(audio_path, sr=22050)
    S = compute_mel_spectrogram(y, sr=sr, n_mels=128, hop_length=512)
    Sn = normalize_spectrogram(S).astype(np.float32)
    n_frames = Sn.shape[1]
    center = n_frames // 2
    half = window_frames // 2
    start = max(0, center-half)
    end = min(n_frames, center+half)
    patch = Sn[:, start:end]
    if patch.shape[1] < window_frames:
        pad = window_frames - patch.shape[1]
        patch = np.pad(patch, ((0,0),(0,pad)), mode='constant', constant_values=0)
    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,window_frames)
    # load model
    model = SmallSpecCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    target_layer = find_last_conv(model)
    if target_layer is None:
        target_layer = model.features[-1]
    gcam = GradCAM(model, target_layer)
    heatmap, pred_class, conf = gcam(x)
    overlay_and_save(S, heatmap, out_png)
    return pred_class, conf

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_classes', type=int, default=25)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pred, conf = run_xai_for_audio(args.audio, args.ckpt, args.out, n_classes=args.n_classes)
    print(f"Pred: {pred}, conf: {conf:.3f}")
