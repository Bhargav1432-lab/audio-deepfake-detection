"""
evaluate_baseline_on_2021_eer.py

Same as evaluate_baseline_on_2021.py, but ALSO saves a per-sample score
for the baseline AASIST so EER can be computed.
Does not overwrite your old results: writes baseline_2021_results_scores.csv

Usage:
    python evaluate_baseline_on_2021_eer.py
"""

import json
import numpy as np
import pandas as pd
import librosa
import torch
from importlib import import_module

CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
MANIFEST_PATH = "feature_cache_manifest_eval.csv"
OUTPUT_CSV = "baseline_2021_results_scores.csv"
NB_SAMP = 64600


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
    audio, _ = librosa.load(filepath, sr=16000)
    audio = pad_or_trim(audio.astype(np.float32))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, output = model(audio_tensor)
        pred = output.argmax(dim=1).item()  # 1 = bonafide, 0 = spoof
        score = (output[0, 1] - output[0, 0]).item()
    return pred, score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_aasist(device)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Total samples: {len(manifest_df)}")

    correct = 0
    total = 0
    errors = 0
    results = []

    for idx, row in manifest_df.iterrows():
        true_is_bonafide = 1 if row["label"] == "bonafide" else 0

        try:
            pred, score = predict_single(model, row["original_filepath"], device)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {row['original_filepath']}: {e}")
            continue

        is_correct = (pred == true_is_bonafide)
        correct += int(is_correct)
        total += 1

        result_row = {
            "filepath": row["original_filepath"],
            "label": row["label"],
            "baseline_pred": pred,
            "baseline_score": score,
            "baseline_correct": is_correct,
        }
        if "codec" in row:
            result_row["codec"] = row["codec"]
            result_row["channel"] = row["channel"]
        results.append(result_row)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(manifest_df)} done")

    accuracy = correct / total if total > 0 else 0
    print(f"\nBaseline accuracy: {accuracy:.4f} ({correct}/{total}, {errors} errors)")
    print("(This should match your earlier baseline number: ~0.906)")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
