"""
cache_features.py

Pre-computes all 4 feature branches (magnitude, phase, residual, prosody)
for every file in your training dataset, and saves them to disk as .npz
files. This is essential because prosody extraction is slow (~15-30 sec
per file) -- without caching, every training epoch would re-extract
features from scratch, making training impractically slow.

After running this once, training scripts load pre-computed features
directly from disk (fast), instead of calling librosa on every batch.

Usage:
    python cache_features.py --split train --limit 2000
    python cache_features.py --split dev --limit 500

--limit caps how many files to process (recommended for your GPU/time
budget -- you don't need all 25,000+ training files for a first version).
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from feature_extractors import extract_all_features

CACHE_DIR = "feature_cache"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="attribution_dataset.csv",
                        help="Source CSV with filepath and label columns")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to process (must match a 'split' column value)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max number of files to process (per class, roughly)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "split" in df.columns and args.split in df["split"].unique():
        df = df[df["split"] == args.split].reset_index(drop=True)

    # normalize label to bonafide/spoof
    # (2019 attribution CSV uses A01 etc; 2021 CSV already uses "spoof" directly)
    df["binary_label"] = df["label"].apply(
        lambda x: "bonafide" if x == "bonafide" else "spoof")

    # balance: take up to `limit` bonafide and `limit` spoof samples
    bonafide_df = df[df["binary_label"] == "bonafide"].head(args.limit)
    spoof_df = df[df["binary_label"] == "spoof"].head(args.limit)
    subset = pd.concat([bonafide_df, spoof_df]).reset_index(drop=True)

    print(f"Processing {len(subset)} files ({len(bonafide_df)} bonafide, {len(spoof_df)} spoof)")
    print(f"Cache directory: {CACHE_DIR}")

    manifest_rows = []
    start_time = time.time()
    errors = 0

    for idx, row in subset.iterrows():
        cache_filename = f"{args.split}_{idx:05d}.npz"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_path):
            row_data = {
                "cache_path": cache_path,
                "label": row["binary_label"],
                "original_filepath": row["filepath"],
            }
            if "codec" in row:
                row_data["codec"] = row["codec"]
                row_data["channel"] = row["channel"]
            manifest_rows.append(row_data)
            continue  # already cached, skip re-processing

        try:
            features = extract_all_features(row["filepath"])
            np.savez(cache_path,
                     magnitude=features["magnitude"],
                     phase=features["phase"],
                     residual=features["residual"],
                     prosody=features["prosody"])
            row_data = {
                "cache_path": cache_path,
                "label": row["binary_label"],
                "original_filepath": row["filepath"],
            }
            if "codec" in row:
                row_data["codec"] = row["codec"]
                row_data["channel"] = row["channel"]
            manifest_rows.append(row_data)
        except Exception as e:
            print(f"  ERROR on {row['filepath']}: {e}")
            errors += 1
            continue

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(subset) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{len(subset)} done "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = f"feature_cache_manifest_{args.split}.csv"
    manifest_df.to_csv(manifest_path, index=False)

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min).")
    print(f"Successfully cached: {len(manifest_df)} | Errors: {errors}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
