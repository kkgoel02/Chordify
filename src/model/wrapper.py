# src/models/wrapper.py
import torch
from .cnn_model import SmallSpecCNN
import os

class ModelWrapper:
    def __init__(self, n_classes, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SmallSpecCNN(n_classes=n_classes).to(self.device)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path, map_location=None):
        map_location = map_location or self.device
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.to(self.device)
