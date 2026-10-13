"""
scan_cache_inf.py

Scans every cached feature file listed in the train, dev and eval
manifests and reports which ones contain NaN or inf values, and in
which feature array (magnitude / phase / residual / prosody).

Also saves the list to bad_cache_files.csv.

Usage:
    python scan_cache_inf.py
"""

import numpy as np
import pandas as pd

MANIFESTS = {
    "train": "feature_cache_manifest_train.csv",
    "dev": "feature_cache_manifest_dev.csv",
    "eval2021": "feature_cache_manifest_eval.csv",
}

bad_rows = []

for split, mpath in MANIFESTS.items():
    df = pd.read_csv(mpath)
    print(f"\n=== {split}: {len(df)} files ===")
    per_key = {}
    n_bad_files = 0

    for _, row in df.iterrows():
        path = row["cache_path"]
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"  could not read {path}: {e}")
            continue

        bad_keys = []
        for key in data.files:
            a = data[key]
            if a.dtype.kind != "f":
                continue
            if not np.isfinite(a).all():
                bad_keys.append(key)
                per_key[key] = per_key.get(key, 0) + 1

        if bad_keys:
            n_bad_files += 1
            bad_rows.append({
                "split": split,
                "cache_path": path,
                "original_filepath": row.get("original_filepath", ""),
                "label": row.get("label", ""),
                "codec": row.get("codec", ""),
                "bad_arrays": ",".join(bad_keys),
            })

    print(f"  files with NaN/inf: {n_bad_files}")
    for key, count in per_key.items():
        print(f"    array '{key}': {count} files")

out = pd.DataFrame(bad_rows)
out.to_csv("bad_cache_files.csv", index=False)
print(f"\nSaved {len(out)} bad files to bad_cache_files.csv")
