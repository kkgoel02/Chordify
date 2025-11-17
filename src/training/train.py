# src/training/train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.models.wrapper import ModelWrapper
from tqdm import tqdm
import os

def simple_train(dataset, n_classes, epochs=3, batch_size=32, lr=1e-3, ckpt_path='checkpoints/model.pt'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    wrapper = ModelWrapper(n_classes=n_classes)
    model = wrapper.model
    device = wrapper.device
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        tot_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        for x, y, *_ in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            pbar.set_postfix(loss=tot_loss/(pbar.n+1))
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    wrapper.save(ckpt_path)
    return ckpt_path
