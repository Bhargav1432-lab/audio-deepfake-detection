"""
cache_indic_features.py

Computes the 4 feature branches (magnitude, phase, residual, prosody) for every
file listed in indic_data/indic_files.csv and saves them as .npz files, using
several CPU processes in parallel (prosody / pYIN is the slow part).

 - Resumable: files that already have a cache file are skipped, so you can stop
   with Ctrl+C and run it again.
 - Includes the residual overflow guard: if the residual array comes out with
   inf/NaN (the problem found earlier in the 2021 eval cache), it is recomputed
   with safe_residual() from fix_bad_residuals.py.
 - Writes feature_cache_manifest_indic.csv, in the same format your other
   manifests use (cache_path, label, original_filepath) plus extra columns.

Usage:
    python cache_indic_features.py --smoke      (uses indic_data_smoke)
    python cache_indic_features.py              (uses indic_data)
    python cache_indic_features.py --workers 4
"""

import os

# one thread per process, otherwise parallel workers fight over the CPU
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import time

import librosa
import numpy as np
import pandas as pd

from feature_extractors import SR, extract_all_features, pad_or_trim_audio
from fix_bad_residuals import safe_residual

CACHE_DIR = "feature_cache_indic"


def work(job):
    file_id, path, cache_path = job
    if os.path.exists(cache_path):
        return file_id, "skip", ""
    try:
        feats = extract_all_features(path)
        notes = []
        for key in list(feats.keys()):
            arr = feats[key]
            if np.isfinite(arr).all():
                continue
            if key == "residual":
                audio, _ = librosa.load(path, sr=SR)
                audio = pad_or_trim_audio(audio)
                res, skipped, n_frames = safe_residual(audio)
                feats["residual"] = res
                notes.append(f"residual repaired ({skipped}/{n_frames} frames skipped)")
            else:
                feats[key] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                notes.append(f"{key} had non-finite values (replaced by 0)")
        tmp = cache_path[:-4] + "_tmp.npz"
        np.savez(tmp, magnitude=feats["magnitude"], phase=feats["phase"],
                 residual=feats["residual"], prosody=feats["prosody"])
        os.replace(tmp, cache_path)
        return file_id, "ok", "; ".join(notes)
    except Exception as e:
        return file_id, "error", str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    data_dir = args.data_dir or ("indic_data_smoke" if args.smoke else "indic_data")
    cache_dir = CACHE_DIR + ("_smoke" if args.smoke else "")
    manifest_out = "feature_cache_manifest_indic" + ("_smoke" if args.smoke else "") + ".csv"

    files = pd.read_csv(os.path.join(data_dir, "indic_files.csv"))
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(r.file_id, r.filepath, os.path.join(cache_dir, f"{r.file_id}.npz"))
            for r in files.itertuples()]
    todo = sum(not os.path.exists(j[2]) for j in jobs)
    print(f"{len(files)} files, {todo} still to process, {args.workers} workers")
    print(f"(rough time: files x ~20 s / workers = about {todo * 20 / args.workers / 60:.0f} min; "
          f"prosody is slow, and each worker also needs a one-off start-up of a minute or so)")

    t0, done, errors, repaired = time.time(), 0, [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=40) as pool:
        for file_id, status, note in pool.imap_unordered(work, jobs, chunksize=1):
            if status != "skip":
                done += 1
            if status == "error":
                errors.append((file_id, note))
                print(f"  ERROR {file_id}: {note}")
            elif note:
                repaired += 1
                print(f"  {file_id}: {note}")
            if status != "skip" and done % 20 == 0:
                el = time.time() - t0
                eta = (todo - done) * el / done
                print(f"  {done}/{todo} done  ({el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left)")

    ok_rows = []
    for r in files.itertuples():
        cp = os.path.join(cache_dir, f"{r.file_id}.npz")
        if os.path.exists(cp):
            d = r._asdict()
            d.pop("Index", None)
            ok_rows.append({"cache_path": cp, "original_filepath": r.filepath,
                            "label": r.label, **{k: v for k, v in d.items()
                                                 if k not in ("filepath", "label")}})
    out = pd.DataFrame(ok_rows)
    out.to_csv(manifest_out, index=False)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. "
          f"Cached: {len(out)}/{len(files)}   errors: {len(errors)}   repaired/modified: {repaired}")
    print(f"Manifest: {manifest_out}")
    print(out.groupby(["language", "kind"]).size().to_string())


if __name__ == "__main__":
    mp.freeze_support()
    main()
