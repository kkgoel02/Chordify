# src/annotations/parser.py
import os

def read_lab_file(path):
    """
    Read a lab file with lines: <start> <end> <label>
    Returns list of (start, end, label).
    """
    intervals = []
    if not os.path.exists(path):
        return intervals
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                s = float(parts[0])
                e = float(parts[1])
            except:
                # if times are malformed skip line
                continue
            label = parts[2]
            intervals.append((s, e, label))
    return intervals

def merge_annotation_files(paths, priority=None):
    """
    Merge the four lab annotation files for one track.
    Simple strategy: concatenate intervals from given paths, sort by start time.
    `paths` is a list of file paths (majmin, majmin7, majmin7inv, majmininv).
    """
    intervals = []
    for p in paths:
        if p and os.path.exists(p):
            intervals.extend(read_lab_file(p))
    intervals.sort(key=lambda x: (x[0], x[1]))
    return intervals

def labels_to_frame_sequence(intervals, sr, hop, n_frames, default_label='N'):
    """
    Convert intervals (start,end,label) -> frame-level label list of length n_frames.
    sr, hop are in samples normally, but for CSV chroma we use sr=1, hop=1 (frame index space).
    """
    seq = [default_label] * n_frames
    if n_frames <= 0:
        return seq
    for s, e, lbl in intervals:
        # when used with chroma CSV (frame units), s and e are frame indices ideally.
        # Our dataset's intervals are in seconds originally; we assume chroma frames index evenly.
        # For CSV metadata we will treat s/e as frame indices when they are integers; otherwise map via sr/hop.
        try:
            # if sr and hop correspond to samples, compute frames:
            start_frame = int((s * sr) / hop)
            end_frame = int((e * sr) / hop)
        except Exception:
            start_frame = int(s)
            end_frame = int(e)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        for f in range(start_frame, end_frame):
            seq[f] = lbl
    return seq
