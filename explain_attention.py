"""
explain_attention.py

NOVELTY 3: Explainability -- attention heatmaps

Uses PyTorch forward hooks to capture the internal attention maps AASIST
computes in its GAT_layer_S (spectral attention) and GAT_layer_T (temporal
attention) layers, WITHOUT modifying AASIST's source code at all.

For a given audio file, this produces:
  1. The model's prediction (real/fake) and confidence
  2. A spectral attention heatmap -- which frequency-region "nodes" the
     model paid most attention to
  3. A temporal attention heatmap -- which time-region "nodes" the model
     paid most attention to

Saves plots as PNG files you can include directly in your report.

Requires: matplotlib (pip install matplotlib if missing)

Usage:
    python explain_attention.py --file "LA/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac"

If no --file given, picks a random bonafide and a random spoof sample
from your attribution dataset automatically.
"""

import argparse
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from importlib import import_module

# ---------------- CONFIG ----------------
CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
NB_SAMP = 64600
OUTPUT_DIR = "explainability_output"
# -----------------------------------------

# Global storage for captured attention maps
captured = {}


def hook_spectral(module, input, output):
    # Recompute att_map the same way forward() does, since forward() doesn't return it.
    x = input[0]
    att_map = module._derive_att_map(x)
    captured["spectral_att"] = att_map.detach().cpu().numpy()


def hook_temporal(module, input, output):
    x = input[0]
    att_map = module._derive_att_map(x)
    captured["temporal_att"] = att_map.detach().cpu().numpy()


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


def analyze_file(model, filepath, device, label_hint=""):
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_padded = pad_or_trim(audio.astype(np.float32))
    audio_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0).to(device)

    captured.clear()
    with torch.no_grad():
        _, output = model(audio_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = output.argmax(dim=1).item()

    pred_label = "BONAFIDE (real)" if pred == 1 else "SPOOF (fake)"
    confidence = probs[pred]

    print(f"\nFile: {filepath}")
    print(f"True label hint: {label_hint}")
    print(f"Prediction: {pred_label} (confidence: {confidence:.4f})")

    # spectral attention: average how much attention each node RECEIVES
    spec_att = captured["spectral_att"][0, :, :, 0]  # (#node, #node)
    spec_importance = spec_att.mean(axis=0)  # avg attention received per node

    temp_att = captured["temporal_att"][0, :, :, 0]
    temp_importance = temp_att.mean(axis=0)

    return {
        "filepath": filepath,
        "pred_label": pred_label,
        "confidence": confidence,
        "spectral_importance": spec_importance,
        "temporal_importance": temp_importance,
        "audio": audio_padded,
        "sr": sr,
    }


def plot_result(result, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # waveform
    axes[0].plot(result["audio"], linewidth=0.5, color="steelblue")
    axes[0].set_title(f"Waveform | Prediction: {result['pred_label']} "
                       f"(confidence: {result['confidence']:.3f})")
    axes[0].set_xlabel("Sample")

    # spectral attention (which frequency-band nodes mattered most)
    axes[1].bar(range(len(result["spectral_importance"])),
                result["spectral_importance"], color="darkorange")
    axes[1].set_title("Spectral Attention (frequency-region importance)")
    axes[1].set_xlabel("Frequency-band node index")
    axes[1].set_ylabel("Attention received")

    # temporal attention (which time-region nodes mattered most)
    axes[2].bar(range(len(result["temporal_importance"])),
                result["temporal_importance"], color="seagreen")
    axes[2].set_title("Temporal Attention (time-region importance)")
    axes[2].set_xlabel("Time-region node index")
    axes[2].set_ylabel("Attention received")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a specific audio file to analyze")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading AASIST model and registering hooks...")
    model = load_aasist(device)
    model.GAT_layer_S.register_forward_hook(hook_spectral)
    model.GAT_layer_T.register_forward_hook(hook_temporal)

    if args.file:
        result = analyze_file(model, args.file, device)
        save_path = f"{OUTPUT_DIR}/explanation_custom.png"
        plot_result(result, save_path)
    else:
        print("No --file given, picking one bonafide and one spoof sample automatically...")
        df = pd.read_csv("attribution_dataset.csv")
        bonafide_sample = df[df["label"] == "bonafide"].sample(1, random_state=42).iloc[0]
        spoof_sample = df[df["label"] != "bonafide"].sample(1, random_state=42).iloc[0]

        result_b = analyze_file(model, bonafide_sample["filepath"], device,
                                 label_hint="bonafide")
        plot_result(result_b, f"{OUTPUT_DIR}/explanation_bonafide_example.png")

        result_s = analyze_file(model, spoof_sample["filepath"], device,
                                 label_hint=spoof_sample["label"])
        plot_result(result_s, f"{OUTPUT_DIR}/explanation_spoof_example.png")

    print(f"\nAll explanation plots saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
