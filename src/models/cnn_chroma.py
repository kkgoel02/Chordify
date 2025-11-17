# src/models/cnn_chroma.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallChromaCNN(nn.Module):
    """
    Small 2D CNN for chroma input shaped (B,1,12,window_frames)
    """
    def __init__(self, n_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,5), padding=(1,2)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,5), padding=(1,2)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        feat = self.conv_block(x)
        out = self.classifier(feat)
        return out
