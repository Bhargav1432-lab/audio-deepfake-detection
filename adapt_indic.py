"""
adapt_indic.py

Few-shot adaptation and cross-lingual transfer on Tamil / Telugu.

Starts from the English-trained checkpoints (ablation_v2/<variant>_seed<seed>.pth)
and fine-tunes them on a handful of Tamil and/or Telugu clips, then tests on
HELD-OUT SPEAKERS. Settings, for every variant and seed:

    zero-shot            no fine-tuning (n = 0)
    train Tamil          fine-tune on n Tamil clips,  test on Tamil AND Telugu
    train Telugu         fine-tune on n Telugu clips, test on Tamil AND Telugu
    train Both           fine-tune on n clips from both languages

Important design choices
  * Only the AAC-matched condition is used (real vs fake_aac), for training and
    testing. With raw fakes, fine-tuning would learn "AAC = real" and the result
    would be inflated.
  * Speaker-disjoint split: within each language and class, half of the speakers
    are used only for training, the other half only for testing. (Real clips come
    from about 20 speakers per language, so the test set is small and noisy.)
  * Training clips are class balanced (half real, half fake, fakes split equally
    over generators). Hyper-parameters (lr, epochs) are fixed and NOT tuned on the
    test speakers.

Usage:
    python adapt_indic.py
    python adapt_indic.py --variants full_adaptive single_phase --sizes 25 100
Output: adapt_results_long.csv and printed tables (mean +- std over the 3 seeds).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

from ablation_v2 import VARIANTS, AblationModel

MANIFEST = "feature_cache_manifest_indic.csv"
BRANCHES = ["magnitude", "phase", "residual", "prosody"]
LANGS = ["Tamil", "Telugu"]
ENTROPY_WEIGHT = 0.5
DEFAULT_VARIANTS = ["full_adaptive", "full_fixed", "drop_residual", "single_phase"]


def eer_pct(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2 or not np.isfinite(s).all():
        return np.nan
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def load_pool():
    meta = pd.read_csv(MANIFEST)
    meta = meta[meta["kind"].isin(["real", "fake_aac"])].reset_index(drop=True)
    meta["spk_key"] = meta["kind"] + "_" + meta["speaker"].astype(str)
    rs = np.random.RandomState(0)           # fixed split, shared with attribution_indic.py
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


def sample_train(meta, langs, n, rng):
    pool = meta[(meta["split"] == "train") & meta["language"].isin(langs)]
    real = pool[pool["kind"] == "real"]
    fake = pool[pool["kind"] == "fake_aac"]
    n_real = n // 2
    n_fake = n - n_real
    gens = sorted(fake["generator"].unique())
    parts = [real.sample(min(n_real, len(real)), random_state=int(rng.integers(1_000_000)))]
    for i, g in enumerate(gens):
        k = n_fake // len(gens) + (1 if i < n_fake % len(gens) else 0)
        fg = fake[fake["generator"] == g]
        parts.append(fg.sample(min(k, len(fg)), random_state=int(rng.integers(1_000_000))))
    return pd.concat(parts).index.values


def make_model(variant, state, device):
    branches, mode = VARIANTS[variant]
    model = AblationModel(branches, mode).to(device)
    model.load_state_dict(state)
    return model


def finetune(model, feats, y, idx, epochs, lr, device, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    idx = torch.as_tensor(np.array(idx))
    model.train()
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(perm), 8):
            b = perm[i:i + 8]
            if len(b) < 2:
                continue
            inputs = {k: feats[k][b].to(device) for k in BRANCHES}
            labels = y[b].to(device)
            if model.gated:
                out, w = model(inputs, return_weights=True)
                ent = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
                loss = ce(out, labels) - ENTROPY_WEIGHT * ent
            else:
                out = model(inputs)
                loss = ce(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def scores_for(model, feats, idx, device):
    model.eval()
    idx = torch.as_tensor(np.array(idx))
    out = []
    for i in range(0, len(idx), 16):
        b = idx[i:i + 16]
        o = model({k: feats[k][b].to(device) for k in BRANCHES})
        out.append((o[:, 1] - o[:, 0]).cpu().numpy())
    return np.concatenate(out)


def summarise(res):
    res = res.copy()
    res["setting"] = res["train"].map(lambda t: "zero-shot" if t == "-" else f"train {t}")
    g = res.groupby(["variant", "setting", "test", "n"])["eer"].agg(["mean", "std"]).reset_index()
    g["txt"] = g.apply(lambda r: f"{r['mean']:5.1f}±{(0 if np.isnan(r['std']) else r['std']):4.1f}", axis=1)
    g["col"] = g["test"] + " n=" + g["n"].astype(str)
    pt = g.pivot_table(index=["variant", "setting"], columns="col", values="txt", aggfunc="first")
    cols = [f"{t} n={n}" for t in LANGS for n in sorted(g["n"].unique()) if f"{t} n={n}" in pt.columns]
    variants = list(dict.fromkeys(res["variant"]))
    settings = ["zero-shot", "train Tamil", "train Telugu", "train Both"]
    rows = [(v, st) for v in variants for st in settings if (v, st) in pt.index]
    return pt.loc[rows, cols].fillna("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 1234])
    ap.add_argument("--sizes", nargs="+", type=int, default=[25, 100])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-dir", default="ablation_v2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_pool()
    print(f"Device: {device}   clips used (real + fake_aac): {len(meta)}")
    print(meta.groupby(["language", "split", "kind"]).size().unstack().to_string())
    feats = load_feats(meta)
    y_all = torch.tensor(np.array((meta["label"] == "bonafide").astype(np.int64)), dtype=torch.long)
    test_idx = {l: meta[(meta["split"] == "test") & (meta["language"] == l)].index.values.copy() for l in LANGS}

    rows = []
    t0 = time.time()
    for variant in args.variants:
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f"{variant}_seed{seed}.pth")
            if not os.path.exists(ckpt):
                print(f"missing {ckpt}, skipped")
                continue
            state = torch.load(ckpt, map_location="cpu")
            model = make_model(variant, state, device)
            for tl in LANGS:                                    # zero-shot
                s = scores_for(model, feats, test_idx[tl], device)
                rows.append(dict(variant=variant, seed=seed, train="-", test=tl, n=0,
                                 eer=eer_pct(y_all[test_idx[tl]].numpy(), s)))
            for train_name, train_langs in [("Tamil", ["Tamil"]), ("Telugu", ["Telugu"]), ("Both", LANGS)]:
                for n in args.sizes:
                    rng = np.random.default_rng(seed * 1000 + n)
                    tr = sample_train(meta, train_langs, n, rng)
                    model = make_model(variant, state, device)
                    finetune(model, feats, y_all, tr, args.epochs, args.lr, device, seed)
                    for tl in LANGS:
                        s = scores_for(model, feats, test_idx[tl], device)
                        rows.append(dict(variant=variant, seed=seed, train=train_name, test=tl, n=n,
                                         eer=eer_pct(y_all[test_idx[tl]].numpy(), s)))
            print(f"  done {variant} seed {seed}  ({(time.time() - t0) / 60:.1f} min)")
            pd.DataFrame(rows).to_csv("adapt_results_long.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv("adapt_results_long.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Test EER (%) on held-out speakers, AAC-matched condition: mean ± std over seeds ===")
    print("(columns: test language and number of fine-tuning clips n; n=0 is zero-shot)")
    print(summarise(res).to_string())
    print("\nSaved adapt_results_long.csv")


if __name__ == "__main__":
    main()
