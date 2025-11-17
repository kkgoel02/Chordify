# src/training/train_chroma.py
import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.chroma_loader import ChromaWindowDataset
from src.models.cnn_chroma import SmallChromaCNN

DEFAULT_WINDOW = 64

def train_main(track_ids=None, epochs=5, batch_size=64, window_frames=DEFAULT_WINDOW, ckpt_path="checkpoints/chroma_cnn.pt"):
    if not os.path.exists("label_map.json"):
        raise FileNotFoundError("label_map.json missing. Run scripts/build_label_map.py first.")

    with open("label_map.json", "r", encoding="utf-8") as fh:
        label_map = json.load(fh)
    n_classes = len(label_map)
    print(f"[INFO] Loaded {n_classes} labels.")

    base = "data/raw/metadata"
    # gather track ids automatically if not provided
    if track_ids is None:
        track_ids = []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name, "bothchroma.csv")
                if os.path.exists(p):
                    track_ids.append(name)
    print(f"[INFO] Training on {len(track_ids)} tracks (sample: {track_ids[:8]})")

    dataset = ChromaWindowDataset(track_ids, base_path="data/raw", label_map=label_map, window_frames=window_frames)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check data/raw/metadata and data/raw/annotations alignment.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallChromaCNN(n_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        for x, y, *_ in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss/(pbar.n+1))
        print(f"[INFO] Epoch {ep+1} avg loss: {epoch_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    train_main()
