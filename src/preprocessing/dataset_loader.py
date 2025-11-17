import os
import pandas as pd
import numpy as np

def load_song_features(song_id, chord_version="majmin"):
    """
    Loads chroma features and chord labels for a given song ID.
    """
    chroma_path = f"data/raw/metadata/{song_id}/bothchroma.csv"
    label_path = f"data/raw/annotations/{song_id}/{chord_version}.lab"

    # Validate paths
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"Chroma file not found: {chroma_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Chord label file not found: {label_path}")

    chroma = pd.read_csv(chroma_path, header=None).to_numpy()

    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines() if line.strip()]

    return chroma, labels


def list_available_songs(base_path="data/raw/annotations"):
    """List all song IDs available in the dataset."""
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])


# -------------------------------
# NEW CODE: DatasetLoader Class
# -------------------------------

class DatasetLoader:
    """
    Handles loading multiple songs from the dataset for training or prototyping.
    """

    def __init__(self, base_path="data/raw", chord_version="majmin"):
        self.base_path = base_path
        self.chord_version = chord_version
        self.annotation_path = os.path.join(base_path, "annotations")
        self.metadata_path = os.path.join(base_path, "metadata")

        # Validate folder existence
        if not os.path.exists(self.annotation_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Dataset structure not found under 'data/raw/'.")
    
    def list_songs(self):
        """Return all available song IDs."""
        return list_available_songs(self.annotation_path)
    
    def load_dataset(self, limit=None):
        """
        Loads multiple songs (up to 'limit' if provided).
        Returns a list of dicts with 'id', 'chroma', and 'labels'.
        """
        song_ids = self.list_songs()
        if limit:
            song_ids = song_ids[:limit]

        dataset = []
        for song_id in song_ids:
            try:
                chroma, labels = load_song_features(song_id, self.chord_version)
                dataset.append({
                    "id": song_id,
                    "chroma": chroma,
                    "labels": labels
                })
            except Exception as e:
                print(f"[WARN] Skipping {song_id}: {e}")

        return dataset
