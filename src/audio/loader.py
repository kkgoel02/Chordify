# src/audio/loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
from .features import load_audio, compute_mel_spectrogram, normalize_spectrogram
from src.annotations.parser import merge_annotation_files, labels_to_frame_sequence
import os

class ChordDataset(Dataset):
    """
    Produces spectrogram windows and center-frame labels.
    audio_paths: list of audio file paths
    annot_paths_map: dict mapping audio_path -> list of annotation file paths
    """
    def __init__(self, audio_paths, annot_paths_map, sr=22050, n_mels=128, n_fft=2048, hop_length=512, window_frames=128, label_map=None):
        self.records = []
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.window_frames = window_frames
        self.label_map = label_map or {}  # map label string -> int

        for ap in audio_paths:
            if not os.path.exists(ap):
                continue
            y, _ = load_audio(ap, sr=self.sr)
            S = compute_mel_spectrogram(y, sr=self.sr, n_mels=self.n_mels, n_fft=n_fft, hop_length=self.hop_length)
            Sn = normalize_spectrogram(S).astype(np.float32)
            n_frames = Sn.shape[1]
            # get annotation files
            ann_paths = annot_paths_map.get(ap, [])
            intervals = merge_annotation_files(ann_paths)
            frame_labels = labels_to_frame_sequence(intervals, self.sr, self.hop_length, n_frames)
            self.records.append({'spec': Sn, 'labels': frame_labels, 'audio_path': ap})

        # build index map (record_idx, center_frame)
        self.index_map = []
        half = self.window_frames // 2
        for ridx, rec in enumerate(self.records):
            n_frames = rec['spec'].shape[1]
            for c in range(half, n_frames - half):
                self.index_map.append((ridx, c))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ridx, c = self.index_map[idx]
        rec = self.records[ridx]
        S = rec['spec']
        labels = rec['labels']
        half = self.window_frames // 2
        start = c - half
        end = c + half
        window = S[:, start:end]
        # pad if necessary
        if window.shape[1] < self.window_frames:
            pad = self.window_frames - window.shape[1]
            window = np.pad(window, ((0,0),(0,pad)), mode='constant', constant_values=0)
        x = torch.from_numpy(window).unsqueeze(0)  # (1, n_mels, window_frames)
        label_str = labels[c] if c < len(labels) else 'N'
        label = self.label_map.get(label_str, -1)
        return x, label, rec['audio_path'], c
