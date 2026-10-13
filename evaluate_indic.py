"""
evaluate_indic.py

Zero-shot cross-lingual test: models trained ONLY on English ASVspoof2019
(all checkpoints in ablation_v2/, 3 seeds each, plus the official AASIST
baseline) are evaluated on Tamil and Telugu without any fine-tuning.

Two conditions per language:
    raw : real speech vs fake speech as released (resampled to 16 kHz)
    aac : real speech vs the same fake clips after an AAC round trip
          (removes the "real = AAC, fake = not" shortcut)
If a model's EER differs a lot between raw and aac, it was partly using
compression artefacts rather than synthesis artefacts.

Outputs (in the current folder):
    indic_scores.csv         one row per file, one column per model
    indic_results_long.csv   EER / AUC / accuracy for every model, language, condition
Prints mean +- std over the 3 seeds, and a per-generator breakdown.

Run it from the aasist folder (the baseline loads config/ and models/ from there).

Usage:
    python evaluate_indic.py --smoke
    python evaluate_indic.py
    python evaluate_indic.py --no-baseline
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from ablation_v2 import VARIANTS, AblationModel
from multiview_model import CachedMultiViewDataset

CONDITIONS = {"raw": ["real", "fake_raw"], "aac": ["real", "fake_aac"]}
FOCUS = ["baseline", "full_adaptive", "full_fixed", "drop_residual", "single_phase", "single_prosody"]


def eer_pct(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2 or not np.isfinite(s).all():
        return np.nan
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def auc_safe(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2 or not np.isfinite(s).all():
        return np.nan
    return roc_auc_score(y, s)


def score_multiview(manifest_csv, ckpts, device):
    models = []
    for path in ckpts:
        m = re.match(r"^(?P<v>.+)_seed(?P<s>\d+)\.pth$", os.path.basename(path))
        variant, seed = m.group("v"), int(m.group("s"))
        if variant not in VARIANTS:
            continue
        branches, mode = VARIANTS[variant]
        net = AblationModel(branches, mode).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        models.append((f"{variant}|{seed}", net))
    print(f"loaded {len(models)} checkpoints")

    loader = DataLoader(CachedMultiViewDataset(manifest_csv), batch_size=8, shuffle=False, num_workers=0)
    scores = {name: [] for name, _ in models}
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            for name, net in models:
                out = net(inputs)
                scores[name].extend((out[:, 1] - out[:, 0]).cpu().numpy())
    return {k: np.array(v) for k, v in scores.items()}


def score_baseline(paths, device):
    import librosa
    from evaluate_baseline_on_2021_eer import load_aasist, pad_or_trim

    model = load_aasist(device)
    out = []
    for i, p in enumerate(paths, 1):
        audio, _ = librosa.load(p, sr=16000)
        x = torch.tensor(pad_or_trim(audio.astype(np.float32)), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, logits = model(x)
        out.append((logits[0, 1] - logits[0, 0]).item())
        if i % 200 == 0:
            print(f"  baseline {i}/{len(paths)}")
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--ckpt-dir", default="ablation_v2")
    args = ap.parse_args()

    manifest_csv = "feature_cache_manifest_indic" + ("_smoke" if args.smoke else "") + ".csv"
    meta = pd.read_csv(manifest_csv).reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   files: {len(meta)}")

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*_seed*.pth")))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}/ (run ablation_v2.py first).")
    scores = score_multiview(manifest_csv, ckpts, device)

    if not args.no_baseline:
        print("scoring baseline AASIST on raw waveforms...")
        scores["baseline|0"] = score_baseline(meta["original_filepath"].tolist(), device)

    pd.concat([meta[["file_id"]], pd.DataFrame(scores)], axis=1).to_csv("indic_scores.csv", index=False)

    y_all = (meta["label"] == "bonafide").astype(int).values
    langs = sorted(meta["language"].unique())
    rows = []
    for key, s in scores.items():
        variant, seed = key.split("|")
        for lang in langs + ["All"]:
            for cond, kinds in CONDITIONS.items():
                mask = np.array(meta["kind"].isin(kinds))  # writable copy
                if lang != "All":
                    mask &= (meta["language"] == lang).values
                y, sc = y_all[mask], s[mask]
                rows.append(dict(variant=variant, seed=int(seed), language=lang, condition=cond,
                                 eer=eer_pct(y, sc), auc=auc_safe(y, sc),
                                 acc0=float(((sc > 0).astype(int) == y).mean()) if len(y) else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv("indic_results_long.csv", index=False)

    order = [v for v in ["baseline"] + list(VARIANTS) if v in set(res["variant"])]

    print("\n=== EER (%), mean over seeds (lower is better) ===")
    piv = res.pivot_table(index="variant", columns=["language", "condition"], values="eer", aggfunc="mean")
    print(piv.reindex(order).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== EER (%), std over seeds ===")
    piv = res.pivot_table(index="variant", columns=["language", "condition"], values="eer", aggfunc="std")
    print(piv.reindex(order).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Accuracy at the default threshold (score > 0), mean over seeds ===")
    piv = res.pivot_table(index="variant", columns=["language", "condition"], values="acc0", aggfunc="mean")
    print(piv.reindex(order).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Per generator: real vs that generator only, raw condition, mean EER (%) ===")
    gens = sorted(g for g in meta["generator"].unique() if g != "real")
    rows = []
    for key, s in scores.items():
        variant, seed = key.split("|")
        if variant not in FOCUS:
            continue
        for lang in langs:
            for g in gens:
                mask = ((meta["language"] == lang) &
                        ((meta["kind"] == "real") | ((meta["kind"] == "fake_raw") & (meta["generator"] == g)))).values
                rows.append(dict(variant=variant, language=lang, generator=g,
                                 eer=eer_pct(y_all[mask], s[mask])))
    pg = pd.DataFrame(rows).pivot_table(index="variant", columns=["language", "generator"], values="eer")
    print(pg.reindex([v for v in FOCUS if v in pg.index]).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nSaved indic_scores.csv and indic_results_long.csv")


if __name__ == "__main__":
    main()
