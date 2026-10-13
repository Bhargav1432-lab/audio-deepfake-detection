"""
test_robustness.py

NOVELTY 4: Real-World Robustness Testing -- Step 2: measure accuracy drop

Runs the baseline AASIST model (real vs fake detector) on the same set of
files across 4 conditions: clean, phone-quality, whatsapp-quality, noisy.
Reports accuracy for each condition so we can see how much performance
degrades under real-world audio conditions.

Requires:
    - degraded_audio_manifest.csv (built by generate_degraded_audio.py)
    - models/weights/AASIST.pth
    - config/AASIST.conf

Usage:
    python test_robustness.py
"""

import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from importlib import import_module

# ---------------- CONFIG ----------------
CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
MANIFEST_PATH = "degraded_audio_manifest.csv"
NB_SAMP = 64600
# -----------------------------------------


def pad_or_trim(audio, target_len=NB_SAMP):
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(target_len / len(audio)) + 1
    return np.tile(audio, num_repeats)[:target_len]


def load_aasist(device):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]

    module = import_module("models.AASIST")
    Model = getattr(module, "Model")
    model = Model(model_config).to(device)
    model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    model.eval()
    return model


def predict_single(model, filepath, device):
    """Returns 1 if predicted bonafide (real), 0 if predicted spoof (fake)."""
    audio, _ = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = pad_or_trim(audio.astype(np.float32))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, output = model(audio_tensor)
        # AASIST output: index 1 = bonafide score (higher = more likely real)
        pred = output.argmax(dim=1).item()
    return pred  # 1 = bonafide, 0 = spoof


def evaluate_condition(model, df, path_column, device):
    correct = 0
    total = 0
    errors = 0

    for _, row in df.iterrows():
        filepath = row[path_column]
        true_label = row["label"]
        true_is_bonafide = 1 if true_label == "bonafide" else 0

        try:
            pred = predict_single(model, filepath, device)
        except Exception as e:
            errors += 1
            continue

        if pred == true_is_bonafide:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, total, errors


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading AASIST model...")
    model = load_aasist(device)

    print("Loading manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Testing on {len(df)} files across 4 conditions\n")

    conditions = [
        ("Clean (original)", "clean_path"),
        ("Phone quality", "phone_path"),
        ("WhatsApp quality", "whatsapp_path"),
        ("Background noise", "noisy_path"),
    ]

    results = {}
    for condition_name, path_col in conditions:
        print(f"Evaluating: {condition_name}...")
        acc, total, errors = evaluate_condition(model, df, path_col, device)
        results[condition_name] = acc
        print(f"  Accuracy: {acc:.4f} ({total} files tested, {errors} errors)\n")

    print("=" * 50)
    print("ROBUSTNESS SUMMARY")
    print("=" * 50)
    clean_acc = results["Clean (original)"]
    for condition_name, acc in results.items():
        drop = clean_acc - acc
        print(f"{condition_name:20s}: {acc:.4f}  (drop from clean: {drop:+.4f})")


if __name__ == "__main__":
    main()
