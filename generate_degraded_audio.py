"""
generate_degraded_audio.py

NOVELTY 4: Real-World Robustness Testing -- Step 1: create degraded audio

Takes a sample of clean audio files from your ASVspoof dataset and creates
3 degraded versions of each, simulating real-world conditions:

    1. phone      - 8kHz, low bitrate, mono (like a phone call)
    2. whatsapp   - heavy compression, low bitrate opus-style artifacts
    3. noisy      - clean audio + added background noise at moderate SNR

Requires: ffmpeg installed and on PATH (confirmed working).

Usage:
    python generate_degraded_audio.py
"""

import os
import subprocess
import pandas as pd
import numpy as np
import soundfile as sf

# ---------------- CONFIG ----------------
CSV_PATH = "attribution_dataset.csv"
NUM_SAMPLES = 100          # how many clean files to degrade (keep small to start)
OUTPUT_DIR = "degraded_audio"
NOISE_SNR_DB = 10          # lower = noisier. 10dB is moderate background noise
SEED = 1234
# -----------------------------------------


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFMPEG ERROR: {result.stderr[-300:]}")
        return False
    return True


def make_phone_quality(input_path, output_path):
    """Simulate phone call: 8kHz sample rate, mono, low bitrate."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "8000", "-ac", "1", "-b:a", "12.2k",
        output_path
    ]
    return run_ffmpeg(cmd)


def make_whatsapp_quality(input_path, output_path):
    """Simulate WhatsApp voice note: opus-style low bitrate compression."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-b:a", "16k",
        output_path
    ]
    return run_ffmpeg(cmd)


def add_background_noise(input_path, output_path, snr_db=NOISE_SNR_DB):
    """Add synthetic white noise at a target SNR level."""
    audio, sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    signal_power = np.mean(audio ** 2)
    noise = np.random.normal(0, 1, len(audio))
    noise_power = np.mean(noise ** 2)

    # scale noise to hit target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))

    noisy_audio = audio + noise
    # avoid clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    sf.write(output_path, noisy_audio, sr)
    return True


def main():
    np.random.seed(SEED)

    df = pd.read_csv(CSV_PATH)
    sample_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)

    for condition in ["clean", "phone", "whatsapp", "noisy"]:
        os.makedirs(os.path.join(OUTPUT_DIR, condition), exist_ok=True)

    manifest_rows = []

    print(f"Processing {len(sample_df)} files into 3 degraded conditions each...")

    for idx, row in sample_df.iterrows():
        src_path = row["filepath"]
        filename = row["filename"]
        label = row["label"]

        if not os.path.exists(src_path):
            print(f"  Skipping missing file: {src_path}")
            continue

        clean_out = os.path.join(OUTPUT_DIR, "clean", f"{filename}.flac")
        phone_out = os.path.join(OUTPUT_DIR, "phone", f"{filename}.wav")
        whatsapp_out = os.path.join(OUTPUT_DIR, "whatsapp", f"{filename}.wav")
        noisy_out = os.path.join(OUTPUT_DIR, "noisy", f"{filename}.wav")

        # copy clean as-is (just for reference / easy comparison)
        if not os.path.exists(clean_out):
            audio, sr = sf.read(src_path)
            sf.write(clean_out, audio, sr)

        ok1 = make_phone_quality(src_path, phone_out)
        ok2 = make_whatsapp_quality(src_path, whatsapp_out)
        ok3 = add_background_noise(src_path, noisy_out)

        if ok1 and ok2 and ok3:
            manifest_rows.append({
                "filename": filename,
                "label": label,
                "clean_path": clean_out,
                "phone_path": phone_out,
                "whatsapp_path": whatsapp_out,
                "noisy_path": noisy_out,
            })

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(sample_df)} done")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv("degraded_audio_manifest.csv", index=False)
    print(f"\nDone. {len(manifest_df)} files successfully processed into 4 conditions.")
    print("Manifest saved to degraded_audio_manifest.csv")


if __name__ == "__main__":
    main()
