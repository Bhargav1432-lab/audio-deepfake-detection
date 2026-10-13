"""
diagnose_domain_shift.py

Why do the English-trained detectors rank REAL Tamil / Telugu speech as more fake
than the synthetic speech (EER above 50%)? One likely reason: recording conditions.
ASVspoof bona fide speech is clean studio audio; Kathbath real speech was recorded
by many people with their own phones, so it is noisier, while the synthetic speech
is clean.

This script measures simple signal properties on random samples from each group
and prints the MEDIAN per group. Nothing is trained and nothing is changed.

Properties (first ~4 s of each clip, 16 kHz, as in the feature pipeline):
  rms_db        overall loudness (dB re full scale)
  noise_floor   10th percentile of frame energy (dB)  -> background noise level
  dyn_range     95th minus 10th percentile of frame energy (dB)
  lead_db       loudness of the first 0.3 s (dB)       -> silence at the start
  hf_frac       share of spectral energy above 4 kHz
  peak          largest absolute sample

Groups: ASVspoof2019 train (bonafide / spoof), ASVspoof2021 eval (bonafide / spoof),
Kathbath real (per language), IndicSynth fake_raw (per generator).

Usage (run from the aasist folder):
    python diagnose_domain_shift.py
    python diagnose_domain_shift.py --n 60
Output: domain_shift_stats.csv and a printed table.
"""

import argparse

import librosa
import numpy as np
import pandas as pd

SR = 16000
N_SAMP = 64600


def stats(path):
    y, _ = librosa.load(path, sr=SR)
    if len(y) == 0:
        return None
    y = y[:N_SAMP]
    eps = 1e-10
    frame, hop = 400, 160
    n_fr = 1 + max(0, (len(y) - frame) // hop)
    fr = np.stack([y[i * hop:i * hop + frame] for i in range(n_fr) if len(y[i * hop:i * hop + frame]) == frame])
    fr_db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + eps)
    lead = y[: int(0.3 * SR)]
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    hf = spec[freqs >= 4000].sum() / (spec.sum() + eps)
    return dict(
        rms_db=20 * np.log10(np.sqrt((y ** 2).mean()) + eps),
        noise_floor=np.percentile(fr_db, 10),
        dyn_range=np.percentile(fr_db, 95) - np.percentile(fr_db, 10),
        lead_db=20 * np.log10(np.sqrt((lead ** 2).mean()) + eps),
        hf_frac=hf,
        peak=np.abs(y).max(),
    )


def sample_rows(df, n, seed):
    return df.sample(min(n, len(df)), random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="clips per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indic", default="indic_data/indic_files.csv")
    ap.add_argument("--train-manifest", default="feature_cache_manifest_train.csv")
    ap.add_argument("--eval-manifest", default="feature_cache_manifest_eval.csv")
    args = ap.parse_args()

    groups = []  # (name, list of paths)
    tr = pd.read_csv(args.train_manifest)
    for lab in ("bonafide", "spoof"):
        groups.append((f"ASVspoof2019 train {lab}",
                       sample_rows(tr[tr["label"] == lab], args.n, args.seed)["original_filepath"].tolist()))
    ev = pd.read_csv(args.eval_manifest)
    for lab in ("bonafide", "spoof"):
        groups.append((f"ASVspoof2021 eval {lab}",
                       sample_rows(ev[ev["label"] == lab], args.n, args.seed)["original_filepath"].tolist()))
    ind = pd.read_csv(args.indic)
    for lang in sorted(ind["language"].unique()):
        sub = ind[(ind["language"] == lang) & (ind["kind"] == "real")]
        groups.append((f"Kathbath real {lang}", sample_rows(sub, args.n, args.seed)["filepath"].tolist()))
    for lang in sorted(ind["language"].unique()):
        for gen in sorted(ind[(ind["language"] == lang) & (ind["kind"] == "fake_raw")]["generator"].unique()):
            sub = ind[(ind["language"] == lang) & (ind["kind"] == "fake_raw") & (ind["generator"] == gen)]
            groups.append((f"IndicSynth fake {lang} {gen}", sample_rows(sub, args.n, args.seed)["filepath"].tolist()))

    rows = []
    for name, paths in groups:
        vals = []
        for p in paths:
            try:
                s = stats(p)
                if s:
                    vals.append(s)
            except Exception as e:
                print(f"  skipped {p}: {type(e).__name__}")
        d = pd.DataFrame(vals)
        med = d.median()
        rows.append(dict(group=name, n=len(d), **med.to_dict()))
        print(f"  measured {name}: {len(d)} clips")

    out = pd.DataFrame(rows)
    out.to_csv("domain_shift_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Median signal properties per group ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print("\nHow to read it: compare the noise_floor column. A much HIGHER (less negative) value for "
          "Kathbath real than for ASVspoof bonafide, with the fake groups LOWER (cleaner), would explain "
          "why an ASVspoof-trained model calls the real clips fake and the clean synthetic clips real.")


if __name__ == "__main__":
    main()
