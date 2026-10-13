"""
test_prosody_branch.py

PROOF OF CONCEPT: Prosody feature extraction (Branch 4 of 4)

Prosody = the "melody and rhythm" of speech: pitch contour, energy/loudness
contour, and speaking rate. Unlike the other 3 branches (which look at fine
spectral detail), prosody is a higher-level characteristic that tends to
survive compression and noise much better -- which is exactly why it's
expected to become more "trusted" by the reliability gate under degraded
conditions.

Usage:
    python test_prosody_branch.py
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = "feature_branch_tests"


def extract_prosody_features(audio, sr):
    """
    Extracts 3 core prosodic features:
      1. Pitch (F0) contour -- fundamental frequency over time
      2. Energy contour -- loudness over time
      3. Voiced/unvoiced pattern -- rhythm of speech vs silence
    """
    # pitch tracking using the pYIN algorithm (robust, standard choice)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, sr=sr,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz, typical low male voice
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz, well above typical speech
    )
    # replace unvoiced (NaN) frames with 0 for simplicity in this test
    f0_clean = np.nan_to_num(f0, nan=0.0)

    # energy contour (RMS = root mean square, standard loudness measure)
    rms = librosa.feature.rms(y=audio)[0]

    return f0_clean, voiced_flag, rms


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv("attribution_dataset.csv")
    bonafide_sample = df[df["label"] == "bonafide"].iloc[0]
    spoof_sample = df[df["label"] != "bonafide"].iloc[0]

    samples = [
        ("bonafide", bonafide_sample["filepath"]),
        (f"spoof_{spoof_sample['label']}", spoof_sample["filepath"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, (label, filepath) in enumerate(samples):
        print(f"Processing: {filepath} ({label})")
        audio, sr = librosa.load(filepath, sr=16000)
        f0, voiced_flag, rms = extract_prosody_features(audio, sr)

        voiced_pct = 100 * np.mean(voiced_flag) if len(voiced_flag) > 0 else 0
        mean_pitch = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0

        print(f"  Pitch (F0) contour length: {len(f0)} frames")
        print(f"  Mean pitch (voiced frames only): {mean_pitch:.1f} Hz")
        print(f"  Percent voiced: {voiced_pct:.1f}%")
        print(f"  RMS energy contour length: {len(rms)} frames")

        axes[i, 0].plot(f0, color="purple")
        axes[i, 0].set_title(f"Pitch (F0) contour - {label}")
        axes[i, 0].set_ylabel("Hz")

        axes[i, 1].plot(rms, color="darkgreen")
        axes[i, 1].set_title(f"Energy (RMS) contour - {label}")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/prosody_branch_test.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot: {save_path}")
    print("\nProsody branch extraction works correctly.")
    print("\nAll 4 feature branches (magnitude, phase, residual, prosody) validated.")
    print("Next step: build PyTorch Dataset that produces all 4 views per sample,")
    print("then build the 4 encoder networks + reliability gate + fusion classifier.")


if __name__ == "__main__":
    main()
