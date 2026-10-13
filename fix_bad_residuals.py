"""
fix_bad_residuals.py

Re-computes ONLY the residual array for the cache files listed in
bad_cache_files.csv (made by scan_cache_inf.py), then writes it back
into the same .npz file. The other three arrays (magnitude, phase,
prosody) are kept exactly as they were, so the slow prosody step is
NOT repeated.

How it differs from extract_residual() in feature_extractors.py:
the algorithm is identical, but a frame is skipped (left as zero, the
same thing the old code did when LPC raised an error) if
  - the LPC coefficients are not finite, or
  - the frame's residual is not finite, or
  - the frame's residual is absurdly large (> MAX_RESIDUAL_ABS).
Originals are backed up to feature_cache_backup_bad/ before overwriting.

Usage:
    python fix_bad_residuals.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import librosa

from feature_extractors import (
    SR, N_FFT, HOP_LENGTH, LPC_ORDER, pad_or_trim_audio, pad_or_trim_2d,
)

MAX_RESIDUAL_ABS = 1e3  # audio is in [-1, 1], so anything near this is a numerical blow-up
BACKUP_DIR = "feature_cache_backup_bad"


def safe_residual(audio):
    frame_length = 400
    hop_length = 160
    residual = np.zeros_like(audio)
    num_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)
    skipped = 0

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        if len(frame) < LPC_ORDER + 1:
            continue
        try:
            lpc_coeffs = librosa.lpc(frame, order=LPC_ORDER)
            if not np.all(np.isfinite(lpc_coeffs)):
                skipped += 1
                continue

            predicted = np.zeros_like(frame)
            for n in range(LPC_ORDER, len(frame)):
                predicted[n] = -np.sum(lpc_coeffs[1:] * frame[n - LPC_ORDER:n][::-1])

            frame_residual = frame - predicted
            if (not np.all(np.isfinite(frame_residual))
                    or np.max(np.abs(frame_residual)) > MAX_RESIDUAL_ABS):
                skipped += 1
                continue

            residual[start:end] += frame_residual
        except Exception:
            skipped += 1
            continue

    stft_result = librosa.stft(residual, n_fft=N_FFT, hop_length=HOP_LENGTH)
    residual_mag = np.abs(stft_result)
    residual_db = librosa.amplitude_to_db(residual_mag, ref=np.max(residual_mag) + 1e-8)
    residual_db = np.nan_to_num(residual_db, nan=0.0, posinf=0.0, neginf=-80.0)
    residual_db = pad_or_trim_2d(residual_db)
    return residual_db.astype(np.float32), skipped, num_frames


def main():
    bad = pd.read_csv("bad_cache_files.csv")
    print(f"Files listed as bad: {len(bad)}")

    only_residual = bad["bad_arrays"] == "residual"
    if not only_residual.all():
        print("WARNING: some files have bad arrays other than residual; skipping those:")
        print(bad[~only_residual][["cache_path", "bad_arrays"]].to_string(index=False))
    bad = bad[only_residual]

    os.makedirs(BACKUP_DIR, exist_ok=True)
    fixed_ok = 0

    for n, (_, row) in enumerate(bad.iterrows(), start=1):
        cache_path = row["cache_path"]
        audio, _ = librosa.load(row["original_filepath"], sr=SR)
        audio = pad_or_trim_audio(audio)

        new_res, skipped, num_frames = safe_residual(audio)

        data = np.load(cache_path, allow_pickle=True)
        arrays = {k: data[k] for k in data.files}
        data.close()

        backup_path = os.path.join(BACKUP_DIR, os.path.basename(cache_path))
        if not os.path.exists(backup_path):
            shutil.copy2(cache_path, backup_path)

        arrays["residual"] = new_res
        np.savez(cache_path, **arrays)

        finite = bool(np.isfinite(new_res).all())
        fixed_ok += int(finite)
        print(f"[{n:2d}/{len(bad)}] {os.path.basename(cache_path)}  codec={row.get('codec', '')}  "
              f"skipped_frames={skipped}/{num_frames}  finite={finite}  "
              f"min={new_res.min():.1f}  max={new_res.max():.1f}")

    print(f"\nFixed and finite: {fixed_ok}/{len(bad)}")
    print("Originals backed up in:", BACKUP_DIR)
    print("Next: python scan_cache_inf.py   (should now report 0 bad files)")


if __name__ == "__main__":
    main()
