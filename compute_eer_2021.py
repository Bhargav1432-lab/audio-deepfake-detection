"""
compute_eer_2021.py

Computes EER (Equal Error Rate, %) for all 3 models on ASVspoof2021 LA:
overall, by codec and by channel.

Run AFTER:
    python evaluate_on_2021_eer.py
    python evaluate_baseline_on_2021_eer.py

Usage:
    python compute_eer_2021.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def eer_safe(y, s):
    if len(set(y)) < 2:  # group has only one class -> EER undefined
        return np.nan
    return compute_eer(y, s)


fus = pd.read_csv("evaluation_2021_results_scores.csv")
base = pd.read_csv("baseline_2021_results_scores.csv")

df = fus.merge(
    base[["filepath", "baseline_score"]],
    left_on="original_filepath", right_on="filepath", how="inner",
)
print(f"Samples used: {len(df)} (fusion file: {len(fus)}, baseline file: {len(base)})")

y = (df["label"] == "bonafide").astype(int).values
models = {
    "Baseline": "baseline_score",
    "Fixed": "fixed_score",
    "Adaptive": "adaptive_score",
}

print("\nOVERALL EER (%)  [lower is better]")
for name, col in models.items():
    eer = eer_safe(y, df[col].values)
    flag = "   <-- above 50%: scores may be pointing the wrong way" if eer > 50 else ""
    print(f"  {name:10s} {eer:.2f}{flag}")

for group in ["codec", "channel"]:
    if group not in df.columns:
        continue
    rows = []
    for g, sub in df.groupby(group):
        ys = (sub["label"] == "bonafide").astype(int).values
        row = {group: g, "n": len(sub)}
        for name, col in models.items():
            row[name] = eer_safe(ys, sub[col].values)
        rows.append(row)
    print(f"\nEER (%) by {group}  (small groups -> noisy)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

df.to_csv("eer_merged_scores.csv", index=False)
print("\nSaved merged scores to eer_merged_scores.csv")
