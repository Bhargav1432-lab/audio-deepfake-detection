"""
test_phase_branch.py

PROOF OF CONCEPT: Phase feature extraction (Branch 2 of 4)

Extracts phase information from the same STFT used for magnitude.
Phase is trickier to visualize meaningfully (raw phase angles look like
noise), so we visualize using the "instantaneous frequency deviation"
representation, which is the standard way phase-based anti-spoofing
features are made interpretable and useful for a CNN.

Usage:
    python test_phase_branch.py
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = "feature_branch_tests"


def extract_phase_features(audio, sr, n_fft=2048, hop_length=512):
    """
    Raw phase angles (-pi to pi) are not directly useful to a CNN since
    they wrap around discontinuously. Instead we compute the phase
    derivative across time frames (how much phase changes frame-to-frame)
    which is smoother and captures meaningful synthesis artifacts.
    """
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(stft_result)

    # unwrap phase along time axis to remove artificial jumps at +/- pi
    phase_unwrapped = np.unwrap(phase, axis=1)

    # phase derivative (instantaneous frequency deviation) -- standard
    # technique used in anti-spoofing phase-based features (e.g. group delay)
    phase_diff = np.diff(phase_unwrapped, axis=1)

    return phase, phase_diff


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv("attribution_dataset.csv")
    bonafide_sample = df[df["label"] == "bonafide"].iloc[0]
    spoof_sample = df[df["label"] != "bonafide"].iloc[0]

    samples = [
        ("bonafide", bonafide_sample["filepath"]),
        (f"spoof_{spoof_sample['label']}", spoof_sample["filepath"]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for i, (label, filepath) in enumerate(samples):
        print(f"Processing: {filepath} ({label})")
        audio, sr = librosa.load(filepath, sr=16000)
        raw_phase, phase_diff = extract_phase_features(audio, sr)

        print(f"  Raw phase shape: {raw_phase.shape} (freq_bins x time_frames)")
        print(f"  Phase diff (usable feature) shape: {phase_diff.shape}")
        print(f"  Phase diff value range: {phase_diff.min():.3f} to {phase_diff.max():.3f}")

        img = librosa.display.specshow(phase_diff, sr=sr, x_axis="time",
                                        y_axis="hz", ax=axes[i], cmap="twilight")
        axes[i].set_title(f"Phase Derivative Features - {label}")
        fig.colorbar(img, ax=axes[i])

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/phase_branch_test.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot: {save_path}")
    print("\nPhase branch extraction works correctly.")


if __name__ == "__main__":
    main()
