"""
build_2021_dataset.py

Reads the ASVspoof2021 LA keys file (trial_metadata.txt) and builds a
clean CSV mapping each audio file to its label AND its channel/codec
metadata -- this is what makes 2021 LA useful for channel-aware
robustness evaluation (unlike 2019 LA, which has no channel variation).

Columns produced:
    filename, filepath, label, system_id, codec, channel, subset

Usage (run from your aasist folder):
    python build_2021_dataset.py
"""

import os
import csv

# ---------------- CONFIG ----------------
KEYS_PATH = "LA2021_keys/LA/CM/trial_metadata.txt"
AUDIO_DIR = "LA2021/flac"
OUTPUT_CSV = "asvspoof2021_dataset.csv"
# -----------------------------------------


def main():
    rows = []

    with open(KEYS_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue  # skip malformed lines

            speaker_id, filename, codec, channel, system_id, label, trim_status, subset = parts

            filepath = os.path.join(AUDIO_DIR, filename + ".flac")
            rows.append({
                "filename": filename,
                "filepath": filepath,
                "label": label,           # "bonafide" or "spoof"
                "system_id": system_id,   # A07-A19, or "-" for bonafide
                "codec": codec,           # e.g. alaw, ulaw, etc.
                "channel": channel,       # e.g. ita_tx, sin_tx, loc_tx
                "subset": subset,
            })

    if not rows:
        print("No rows parsed. Check KEYS_PATH.")
        return

    missing = sum(1 for r in rows if not os.path.exists(r["filepath"]))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "filepath", "label", "system_id", "codec", "channel", "subset"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Missing audio files: {missing}")

    # summary stats
    label_counts = {}
    codec_counts = {}
    channel_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        codec_counts[r["codec"]] = codec_counts.get(r["codec"], 0) + 1
        channel_counts[r["channel"]] = channel_counts.get(r["channel"], 0) + 1

    print("\nLabel distribution:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")

    print("\nCodec distribution:")
    for k, v in sorted(codec_counts.items()):
        print(f"  {k}: {v}")

    print("\nChannel distribution:")
    for k, v in sorted(channel_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
