"""
extra_metrics_2021.py

Run AFTER compute_eer_2021.py (needs eer_merged_scores.csv).

Prints:
  1. Class balance of the 585-sample evaluation set
  2. ROC-AUC, precision, recall, F1 for each model (threshold: score > 0)
  3. Bootstrap 95% confidence intervals for the overall EER, and how often
     Adaptive beats Baseline / Fixed across resamples
  4. A sanity check: do any train/dev files also appear in the eval set?

Usage:
    python extra_metrics_2021.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

df = pd.read_csv("eer_merged_scores.csv")
y = (df["label"] == "bonafide").astype(int).values  # 1 = bonafide, 0 = spoof
models = {
    "Baseline": "baseline_score",
    "Fixed": "fixed_score",
    "Adaptive": "adaptive_score",
}


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


print("=== 1. Evaluation set balance ===")
print(f"bonafide: {int(y.sum())}   spoof: {int((1 - y).sum())}   total: {len(y)}")

print("\n=== 2. ROC-AUC, precision, recall, F1 (threshold: score > 0) ===")
rows = []
for name, col in models.items():
    s = df[col].values
    pred = (s > 0).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    rows.append({
        "model": name,
        "AUC": roc_auc_score(y, s),
        "spoof_P": p[0], "spoof_R": r[0], "spoof_F1": f[0],
        "bona_P": p[1], "bona_R": r[1], "bona_F1": f[1],
    })
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== 3. Bootstrap 95% CI for overall EER (1000 resamples) ===")
rng = np.random.default_rng(0)
n = len(y)
boot = {name: [] for name in models}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if len(set(yb)) < 2:
        continue
    for name, col in models.items():
        boot[name].append(compute_eer(yb, df[col].values[idx]))

for name in models:
    arr = np.array(boot[name])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"  {name:10s} EER = {compute_eer(y, df[models[name]].values):6.2f}%   95% CI [{lo:.2f}, {hi:.2f}]")

ad = np.array(boot["Adaptive"])
for other in ["Baseline", "Fixed"]:
    ot = np.array(boot[other])
    print(f"  Adaptive EER lower than {other} in {100 * np.mean(ad < ot):.1f}% of resamples")

print("\n=== 4. Overlap check: train/dev vs eval ===")


def names(path):
    d = pd.read_csv(path)
    return set(os.path.splitext(os.path.basename(p))[0] for p in d["original_filepath"])


tr = names("feature_cache_manifest_train.csv")
dv = names("feature_cache_manifest_dev.csv")
ev = names("feature_cache_manifest_eval.csv")
print(f"  train files: {len(tr)}   dev files: {len(dv)}   eval files: {len(ev)}")
print(f"  same file name in train AND eval: {len(tr & ev)}")
print(f"  same file name in dev   AND eval: {len(dv & ev)}")
print("  (both should be 0)")
