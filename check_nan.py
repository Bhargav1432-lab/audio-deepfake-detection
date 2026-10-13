"""
check_nan.py

Finds which samples have NaN scores, and whether the cached features
for those samples contain NaN.

Usage:
    python check_nan.py
"""

import numpy as np
import pandas as pd

df = pd.read_csv("evaluation_2021_results_scores.csv")
print(f"Total samples: {len(df)}")

bad_any = pd.Series(False, index=df.index)

print("\n--- NaN scores ---")
for col in ["fixed_score", "adaptive_score"]:
    bad = df[col].isna()
    bad_any |= bad
    print(f"{col}: {bad.sum()} NaN out of {len(df)}")
    if bad.sum():
        print("   by label:", df[bad]["label"].value_counts().to_dict())
        if "codec" in df.columns:
            print("   by codec:", df[bad]["codec"].value_counts().to_dict())
        pred_col = col.replace("score", "pred")
        corr_col = col.replace("score", "correct")
        print("   predicted class:", df[bad][pred_col].value_counts().to_dict())
        print("   fraction counted 'correct':", round(df[bad][corr_col].mean(), 3))

print("\n--- Columns in the results file ---")
print(list(df.columns))

print("\n--- NaN inside cached feature files ---")
npz_cols = []
for c in df.columns:
    vals = df[c].astype(str)
    if vals.str.lower().str.endswith(".npz").all():
        npz_cols.append(c)

if not npz_cols:
    print("No column of .npz paths found. Paste the column list above to me.")
else:
    for c in npz_cols:
        nan_rows = {}
        for i, path in df[c].items():
            try:
                data = np.load(path, allow_pickle=True)
                for key in data.files:
                    arr = data[key]
                    if arr.dtype.kind == "f" and np.isnan(arr).any():
                        nan_rows.setdefault(key, []).append(i)
            except Exception as e:
                print(f"  could not read {path}: {e}")
                break
        print(f"Column '{c}':")
        if not nan_rows:
            print("   no NaN found in any feature array")
        for key, rows in nan_rows.items():
            overlap = bad_any.loc[rows].sum()
            print(f"   array '{key}': {len(rows)} samples contain NaN "
                  f"({overlap} of them are also NaN-score samples)")
