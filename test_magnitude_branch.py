"""
test_magnitude_branch.py

PROOF OF CONCEPT: Magnitude feature extraction (Branch 1 of 4)

Extracts a magnitude spectrogram from a sample audio file and saves a
visualization, to confirm this works correctly on your ASVspoof data
before building it into a full CNN branch.

Usage:
    python test_magnitude_branch.py
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = "feature_branch_tests"


def extract_magnitude(audio, sr, n_fft=2048, hop_length=512):
    """
    Returns the magnitude spectrogram: how much energy is present at
    each frequency, at each point in time.
    """
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_result)
    # convert to decibel scale (log scale) -- standard practice, makes
    # quiet and loud parts both visible
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return magnitude_db


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # grab one bonafide and one spoof sample from your existing dataset
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
        magnitude_db = extract_magnitude(audio, sr)

        print(f"  Audio length: {len(audio)} samples ({len(audio)/sr:.2f} sec)")
        print(f"  Magnitude spectrogram shape: {magnitude_db.shape} (freq_bins x time_frames)")
        print(f"  Value range: {magnitude_db.min():.2f} dB to {magnitude_db.max():.2f} dB")

        img = librosa.display.specshow(magnitude_db, sr=sr, x_axis="time",
                                        y_axis="hz", ax=axes[i], cmap="viridis")
        axes[i].set_title(f"Magnitude Spectrogram - {label}")
        fig.colorbar(img, ax=axes[i], format="%+2.0f dB")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/magnitude_branch_test.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot: {save_path}")
    print("\nMagnitude branch extraction works correctly.")
    print("Next step: wrap this into a proper PyTorch Dataset + small CNN.")


if __name__ == "__main__":
    main()
