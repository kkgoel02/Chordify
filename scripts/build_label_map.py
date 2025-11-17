#!/usr/bin/env python3
# scripts/build_label_map.py
import os
import json

ANNOT_DIR = "data/raw/annotations"
OUT_FILE = "label_map.json"

def extract_labels_from_lab(path):
    labels = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(parts[2])
    return labels

def main():
    if not os.path.isdir(ANNOT_DIR):
        print(f"[ERROR] Annotation directory not found: {ANNOT_DIR}")
        return

    all_labels = set()
    for folder in sorted(os.listdir(ANNOT_DIR)):
        folder_path = os.path.join(ANNOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fn in os.listdir(folder_path):
            if fn.endswith(".lab"):
                p = os.path.join(folder_path, fn)
                all_labels.update(extract_labels_from_lab(p))

    if not all_labels:
        print("[WARN] No labels found in annotations. Check dataset path/format.")
        return

    sorted_labels = sorted(list(all_labels))
    label_map = {lab: idx for idx, lab in enumerate(sorted_labels)}

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(label_map)} labels to {OUT_FILE}")

if __name__ == "__main__":
    main()
