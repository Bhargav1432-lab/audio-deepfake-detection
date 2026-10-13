"""
build_attribution_dataset.py

Reads the ASVspoof2019 LA protocol files (train + dev, which only contain
systems A01-A06 plus bonafide) and builds a clean CSV mapping:

    filename, filepath, label

where label is one of: bonafide, A01, A02, A03, A04, A05, A06

This CSV is what you'll feed into a PyTorch Dataset class later to train
the multi-class "which system generated this" classifier (Novelty 1).

Usage (run from inside your aasist folder, e.g. C:\\Users\\bharg\\aasist):
    python build_attribution_dataset.py
"""

import os
import csv

# ---- CONFIG: adjust these if your folder names differ ----
LA_ROOT = "LA"
PROTOCOL_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols")
TRAIN_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

TRAIN_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac")

OUTPUT_CSV = "attribution_dataset.csv"
# ------------------------------------------------------------


def parse_protocol(protocol_path, audio_dir, split_name):
    """
    Each line looks like:
        LA_0079 LA_T_1138215 - - bonafide
        LA_0069 LA_T_1234567 - A01 spoof

    Columns: SPEAKER_ID  FILENAME  -  SYSTEM_ID  KEY
    """
    rows = []
    if not os.path.exists(protocol_path):
        print(f"WARNING: protocol file not found: {protocol_path}")
        return rows

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            speaker_id, filename, _, system_id, key = parts

            if key == "bonafide":
                label = "bonafide"
            else:
                label = system_id  # e.g. A01, A02, ... A06

            filepath = os.path.join(audio_dir, filename + ".flac")
            rows.append({
                "filename": filename,
                "filepath": filepath,
                "label": label,
                "split": split_name,
            })
    return rows


def main():
    all_rows = []
    all_rows += parse_protocol(TRAIN_PROTOCOL, TRAIN_AUDIO_DIR, "train")
    all_rows += parse_protocol(DEV_PROTOCOL, DEV_AUDIO_DIR, "dev")

    if not all_rows:
        print("No rows parsed. Check your paths at the top of this script.")
        return

    # Sanity check: how many files actually exist on disk?
    missing = 0
    for row in all_rows:
        if not os.path.exists(row["filepath"]):
            missing += 1

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "filepath", "label", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Print label distribution summary
    label_counts = {}
    for row in all_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_CSV}")
    print(f"Missing audio files (path didn't exist): {missing}")
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")


if __name__ == "__main__":
    main()
