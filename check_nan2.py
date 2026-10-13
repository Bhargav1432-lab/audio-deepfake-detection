"""
check_nan2.py

The first check found no NaN inside the cached feature files, yet 36
samples produce NaN scores. This compares the cached features of those
36 samples against normal samples, looking for inf values, all-zero
arrays, constant arrays, and extreme values.

Usage:
    python check_nan2.py
"""

import numpy as np
import pandas as pd

df = pd.read_csv("evaluation_2021_results_scores.csv")
bad = df[df["fixed_score"].isna()]
good = df[~df["fixed_score"].isna()].sample(60, random_state=0)


def collect(sub):
    out = {}
    for path in sub["cache_path"]:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            a = data[key]
            if a.dtype.kind not in "fi":
                continue
            a = a.astype(np.float64)
            d = out.setdefault(
                key,
                {"n": 0, "inf": 0, "allzero": 0, "const": 0, "maxabs": [], "shapes": set()},
            )
            d["n"] += 1
            d["inf"] += int(np.isinf(a).any())
            d["allzero"] += int((a == 0).all())
            d["const"] += int(a.std() == 0)
            finite = a[np.isfinite(a)]
            d["maxabs"].append(np.abs(finite).max() if finite.size else np.nan)
            d["shapes"].add(a.shape)
    return out


for name, sub in [("NaN-score samples (36)", bad), ("normal samples (60 random)", good)]:
    print(f"\n=== {name} ===")
    stats = collect(sub)
    for key, d in stats.items():
        print(
            f"  {key}: n={d['n']}  with_inf={d['inf']}  all_zero={d['allzero']}  "
            f"constant={d['const']}  max|x| median={np.nanmedian(d['maxabs']):.3g} "
            f"largest={np.nanmax(d['maxabs']):.3g}  shapes={sorted(d['shapes'])[:3]}"
        )
