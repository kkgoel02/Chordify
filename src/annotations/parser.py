# src/annotations/parser.py
import os
import numpy as np

def read_lab_file(path):
    """Read simple lab file with lines: start end label"""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            label = " ".join(parts[2:])
            items.append((start, end, label))
    return items

def merge_annotation_files(paths, priority=None):
    """
    Merge several lab annotation lists into a sorted interval list.
    Simple heuristic: concatenate intervals from files in `paths` order (priority),
    then sort and coalesce identical-adjacent labels.
    """
    intervals = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        intervals.extend(read_lab_file(p))
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: (x[0], x[1]))
    # Merge adjacent intervals with same label
    merged = []
    for s,e,l in intervals:
        if not merged:
            merged.append([s,e,l])
        else:
            ps,pe,pl = merged[-1]
            if abs(s - pe) < 1e-6 and pl == l:
                merged[-1][1] = e
            else:
                # if overlapping with different label, we keep both (could refine)
                if s < pe:
                    # split overlapping: keep earlier part assigned to previous label
                    if e <= pe:
                        # fully inside previous interval -> skip or keep depending
                        continue
                    else:
                        # partial overlap: start new after previous end
                        merged.append([pe, e, l])
                else:
                    merged.append([s,e,l])
    return [(float(a), float(b), str(c)) for a,b,c in merged]

def labels_to_frame_sequence(intervals, sr, hop_length, n_frames, default_label='N'):
    """
    Convert intervals [(start,end,label), ...] to a frame-level label array of length n_frames.
    sr: sample rate
    hop_length: spectrogram hop length (samples)
    """
    seq = [default_label] * n_frames
    for s,e,l in intervals:
        start_frame = int(np.floor((s * sr) / hop_length))
        end_frame = int(np.ceil((e * sr) / hop_length))
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        for f in range(start_frame, end_frame):
            seq[f] = l
    return seq
