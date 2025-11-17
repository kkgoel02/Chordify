# src/audio/features.py
import numpy as np
import librosa

def load_audio(path, sr=22050, mono=True):
    y, _sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr

def compute_mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512, fmin=20, fmax=None):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax or sr//2
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S  # shape (n_mels, n_frames)

def normalize_spectrogram(S):
    mean = S.mean()
    std = S.std() if S.std() > 0 else 1.0
    return (S - mean) / std
