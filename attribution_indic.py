"""
attribution_indic.py

Generator attribution ("provenance") on Tamil / Telugu.

Question: does the representation learned by the English-trained detectors carry
information about WHICH system produced a clip? Classes: real, xtts_v2, freevc24
(vits is absent from the Tamil and Telugu data we found).

Method: take the fused embedding of each English-trained checkpoint (frozen, no
fine-tuning), train a small logistic-regression head on it using the TRAIN speakers
of one language, and test on held-out speakers of the same language and of the
other language (cross-lingual attribution). Only the AAC-matched condition is used
(real and fake_aac), so compression cannot give the answer away.

The speaker split is the same as in adapt_indic.py.

Usage:
    python attribution_indic.py
Output: attribution_results_long.csv and printed tables.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from ablation_v2 import VARIANTS, AblationModel

MANIFEST = "feature_cache_manifest_indic.csv"
BRANCHES = ["magnitude", "phase", "residual", "prosody"]
LANGS = ["Tamil", "Telugu"]
DEFAULT_VARIANTS = ["full_adaptive", "full_fixed", "drop_residual", "single_phase"]


def load_pool():
    meta = pd.read_csv(MANIFEST)
    meta = meta[meta["kind"].isin(["real", "fake_aac"])].reset_index(drop=True)
    meta["spk_key"] = meta["kind"] + "_" + meta["speaker"].astype(str)
    rs = np.random.RandomState(0)
    split = {}
    for (lang, kind), g in meta.groupby(["language", "kind"]):
        spk = sorted(g["spk_key"].unique())
        rs.shuffle(spk)
        half = len(spk) // 2
        for i, s in enumerate(spk):
            split[(lang, s)] = "train" if i < half else "test"
    meta["split"] = [split[(l, s)] for l, s in zip(meta["language"], meta["spk_key"])]
    return meta


def load_feats(meta):
    out = {k: [] for k in BRANCHES}
    for p in meta["cache_path"]:
        d = np.load(p)
        for k in BRANCHES:
            out[k].append(d[k])
    return {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}


@torch.no_grad()
def embed(model, feats, device):
    model.eval()
    n = len(feats["prosody"])
    out = []
    for i in range(0, n, 16):
        inputs = {k: feats[k][i:i + 16].to(device) for k in BRANCHES}
        embs = [model.encoders[b](inputs[b]) for b in model.branches]
        if model.mode == "gated":
            w = model.gate(torch.cat(embs, dim=1))
            fused = (torch.stack(embs, dim=1) * w.unsqueeze(-1)).sum(dim=1)
        elif model.mode == "concat":
            fused = torch.cat(embs, dim=1)
        else:
            fused = embs[0]
        out.append(fused.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 1234])
    ap.add_argument("--ckpt-dir", default="ablation_v2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_pool()
    classes = ["real"] + sorted(g for g in meta["generator"].unique() if g != "real")
    y_all = meta["generator"].map({c: i for i, c in enumerate(classes)}).values
    print(f"Device: {device}   classes: {classes}")
    print(meta.groupby(["language", "split", "generator"]).size().unstack().to_string())
    feats = load_feats(meta)

    rows, cms = [], {}
    for variant in args.variants:
        branches, mode = VARIANTS[variant]
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f"{variant}_seed{seed}.pth")
            if not os.path.exists(ckpt):
                print(f"missing {ckpt}, skipped")
                continue
            model = AblationModel(branches, mode).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            emb = embed(model, feats, device)
            for tr_lang in LANGS:
                tr = meta[(meta["split"] == "train") & (meta["language"] == tr_lang)].index.values
                sc = StandardScaler().fit(emb[tr])
                clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(emb[tr]), y_all[tr])
                for te_lang in LANGS:
                    te = meta[(meta["split"] == "test") & (meta["language"] == te_lang)].index.values
                    pred = clf.predict(sc.transform(emb[te]))
                    labels = list(range(len(classes)))
                    rows.append(dict(
                        variant=variant, seed=seed, train=tr_lang, test=te_lang,
                        acc=accuracy_score(y_all[te], pred),
                        bal_acc=balanced_accuracy_score(y_all[te], pred),
                        macro_f1=f1_score(y_all[te], pred, average="macro", labels=labels, zero_division=0),
                        majority=np.bincount(y_all[te], minlength=len(classes)).max() / len(te)))
                    key = (variant, tr_lang, te_lang)
                    cm = confusion_matrix(y_all[te], pred, labels=labels)
                    cms[key] = cms.get(key, 0) + cm
            print(f"  done {variant} seed {seed}")

    res = pd.DataFrame(rows)
    res.to_csv("attribution_results_long.csv", index=False)
    pd.set_option("display.width", 200)
    for metric in ("acc", "bal_acc", "macro_f1"):
        g = res.groupby(["variant", "train", "test"])[metric].agg(["mean", "std"]).reset_index()
        g["txt"] = g.apply(lambda r: f"{r['mean']:.3f}±{(0 if np.isnan(r['std']) else r['std']):.3f}", axis=1)
        g["col"] = "train " + g["train"] + " -> test " + g["test"]
        print(f"\n=== {metric} (mean ± std over seeds) ===")
        print(g.pivot_table(index="variant", columns="col", values="txt", aggfunc="first").to_string())
    maj = res.groupby(["train", "test"])["majority"].mean()
    print("\nmajority-class accuracy (the 'do nothing' level for acc). bal_acc chance = 1 / number of classes:")
    print(maj.round(3).to_string())

    focus = args.variants[0]
    print(f"\n=== Confusion matrices for {focus}, summed over seeds (rows = true, columns = predicted: {classes}) ===")
    for tr_lang in LANGS:
        for te_lang in LANGS:
            if (focus, tr_lang, te_lang) in cms:
                print(f"\ntrain {tr_lang} -> test {te_lang}")
                print(pd.DataFrame(cms[(focus, tr_lang, te_lang)], index=classes, columns=classes).to_string())
    print("\nSaved attribution_results_long.csv")


if __name__ == "__main__":
    main()
