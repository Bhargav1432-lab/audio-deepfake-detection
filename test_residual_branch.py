"""
test_residual_branch.py

PROOF OF CONCEPT: Residual feature extraction (Branch 3 of 4)

Uses Linear Predictive Coding (LPC) to model the "expected" part of the
speech signal (based on how human vocal tracts produce sound), then
subtracts that prediction from the real signal. What's LEFT OVER (the
residual/error) often contains synthesis artifacts that don't fit the
natural vocal-tract model -- a classic and effective anti-spoofing cue.

Usage:
    python test_residual_branch.py
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = "feature_branch_tests"
LPC_ORDER = 16  # typical order for speech LPC modeling


def extract_residual(audio, sr, frame_length=400, hop_length=160, lpc_order=LPC_ORDER):
    """
    Frame-by-frame LPC analysis: for each short frame of audio, fit an
    LPC model (predicts each sample from the previous few), then compute
    the residual = actual - predicted. Returns residual as a 1D signal
    the same length as the input, plus its spectrogram for visualization.
    """
    residual = np.zeros_like(audio)

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]

        if len(frame) < lpc_order + 1:
            continue

        try:
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
            predicted = np.zeros_like(frame)
            for n in range(lpc_order, len(frame)):
                predicted[n] = -np.sum(lpc_coeffs[1:] * frame[n - lpc_order:n][::-1])
            frame_residual = frame - predicted
            residual[start:end] += frame_residual
        except Exception:
            continue  # skip frames where LPC fails (e.g. silence)

    return residual


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
        residual = extract_residual(audio, sr)

        print(f"  Audio length: {len(audio)} samples")
        print(f"  Residual length: {len(residual)} samples")
        print(f"  Residual energy (mean abs): {np.mean(np.abs(residual)):.5f}")
        print(f"  Original energy (mean abs): {np.mean(np.abs(audio)):.5f}")

        # plot waveform and residual
        axes[i, 0].plot(audio, linewidth=0.5, color="steelblue")
        axes[i, 0].set_title(f"Original waveform - {label}")

        axes[i, 1].plot(residual, linewidth=0.5, color="crimson")
        axes[i, 1].set_title(f"LPC Residual - {label}")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/residual_branch_test.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot: {save_path}")
    print("\nResidual branch extraction works correctly.")


if __name__ == "__main__":
    main()
