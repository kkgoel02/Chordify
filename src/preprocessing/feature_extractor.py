import numpy as np

def build_chord_vocabulary(dataset):
    """
    Builds a mapping of all unique chords to integer IDs.
    """
    chords = set()
    for song in dataset:
        for _, _, chord in song['labels']:
            if chord != 'N':  # Ignore 'no chord' segments
                chords.add(chord)
    chord_to_idx = {chord: i for i, chord in enumerate(sorted(chords))}
    idx_to_chord = {i: chord for chord, i in chord_to_idx.items()}
    return chord_to_idx, idx_to_chord


def align_frames_to_chords(song, chord_to_idx, hop_length=0.1):
    """
    Align chroma frames to chord labels using frame duration.
    Parameters:
        song (dict): { 'id', 'chroma', 'labels' }
        chord_to_idx (dict): mapping from chord name to integer index
        hop_length (float): seconds per frame (approx.)
    Returns:
        (np.ndarray, np.ndarray): (X, y) for this song
    """
    chroma = song['chroma']
    labels = song['labels']

    num_frames = chroma.shape[0]
    frame_times = np.arange(num_frames) * hop_length  # time stamps for each frame

    y = np.full(num_frames, -1)  # initialize with -1 (undefined)
    for start, end, chord in labels:
        start, end = float(start), float(end)
        mask = (frame_times >= start) & (frame_times < end)
        if chord != 'N' and chord in chord_to_idx:  # skip 'no chord'
            y[mask] = chord_to_idx[chord]  # assign chord index

    valid_mask = y != -1
    X = chroma[valid_mask]
    y = y[valid_mask]

    return X, y


def prepare_training_data(dataset, hop_length=0.1):
    """
    Converts full dataset into aligned model-ready arrays.
    Returns:
        X (np.ndarray), y (np.ndarray), chord_to_idx (dict), idx_to_chord (dict)
    """
    chord_to_idx, idx_to_chord = build_chord_vocabulary(dataset)

    all_X, all_y = [], []
    for song in dataset:
        X_song, y_song = align_frames_to_chords(song, chord_to_idx, hop_length)
        all_X.append(X_song)
        all_y.append(y_song)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"Final dataset: {X.shape[0]} frames, {X.shape[1]} features, {len(chord_to_idx)} chord classes")

    return X, y, chord_to_idx, idx_to_chord


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def normalize_and_split(X, y, test_size=0.2):
    """
    Normalize features and split dataset into train and test sets.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler
