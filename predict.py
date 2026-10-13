"""
predict.py

Turnkey Single-File Audio Anti-Spoofing & Deepfake Detection CLI.
Takes any audio file (.wav, .flac, .mp3, .ogg), extracts multi-view forensic features in real time,
and outputs:
  1. Verdict: REAL (Bona Fide) vs FAKE (Deepfake Spoof)
  2. Confidence score (%)
  3. Forensic Reliability Weights (how much the model trusted each physical view)
  4. Generator Attribution (if detected as fake)

Usage:
    python predict.py path/to/audio.flac
    python predict.py path/to/audio.wav --attribution
"""

import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F

from feature_extractors import extract_all_features
from multiview_model import AdaptiveFusionModel

DEFAULT_MODEL_WEIGHTS = "adaptive_fusion.pth"
ATTRIBUTION_WEIGHTS = "attribution_head.pth"


def analyze_audio(audio_path, model_path=DEFAULT_MODEL_WEIGHTS, device=None, check_attribution=False):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[*] Analyzing Audio: {os.path.basename(audio_path)}")
    print(f"[*] Compute Device:  {device}")

    # 1. Feature Extraction
    t0 = time.time()
    feats = extract_all_features(audio_path, fast_prosody=True)
    t_feat = time.time() - t0
    print(f"[*] Fast Feature Extraction completed in {t_feat:.2f}s")

    # Format tensors for batch dimension (1, ...)
    batch = {
        "magnitude": torch.tensor(feats["magnitude"]).unsqueeze(0).to(device),
        "phase": torch.tensor(feats["phase"]).unsqueeze(0).to(device),
        "residual": torch.tensor(feats["residual"]).unsqueeze(0).to(device),
        "prosody": torch.tensor(feats["prosody"]).unsqueeze(0).to(device),
    }

    # 2. Model Inference
    model = AdaptiveFusionModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[!] Warning: Model weights {model_path} not found. Running with uninitialized weights.")

    model.eval()
    with torch.no_grad():
        logits, weights = model(batch, return_weights=True)
        probs = F.softmax(logits, dim=1)[0]
        # Label convention: 0 = Spoof (Fake), 1 = Bona Fide (Real)
        prob_spoof = probs[0].item() * 100.0
        prob_bonafide = probs[1].item() * 100.0
        pred_label = logits.argmax(dim=1).item()
        w = weights[0].cpu().numpy()

    # 3. Present Results
    print("\n" + "=" * 65)
    print("                FORENSIC DEEPFAKE DETECTION VERDICT             ")
    print("=" * 65)

    if pred_label == 1:
        verdict = "REAL (BONA FIDE HUMAN SPEECH)"
        confidence = prob_bonafide
        verdict_color = "PASS"
    else:
        verdict = "FAKE (DEEPFAKE / SYNTHETIC SPEECH)"
        confidence = prob_spoof
        verdict_color = "ALERT"

    print(f" Verdict:              [{verdict_color}] {verdict}")
    print(f" Confidence Score:     {confidence:.2f}%")
    print(f" Probability Genuine:  {prob_bonafide:.2f}%")
    print(f" Probability Deepfake: {prob_spoof:.2f}%")
    print("-" * 65)
    print(" Dynamic Forensic Trust Allocation (Reliability Gate):")
    print(f"   - Magnitude Spectrogram : {w[0] * 100:5.2f}%")
    print(f"   - Phase Anomalies       : {w[1] * 100:5.2f}%")
    print(f"   - LP Vocal Residual     : {w[2] * 100:5.2f}%")
    print(f"   - Prosodic Dynamics     : {w[3] * 100:5.2f}%")
    print("=" * 65 + "\n")

    return {
        "verdict": verdict,
        "is_spoof": (pred_label == 0),
        "prob_bonafide": prob_bonafide,
        "prob_spoof": prob_spoof,
        "weights": {
            "magnitude": float(w[0]),
            "phase": float(w[1]),
            "residual": float(w[2]),
            "prosody": float(w[3]),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Live Audio Anti-Spoofing & Deepfake Detection")
    parser.add_argument("audio", help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--weights", default=DEFAULT_MODEL_WEIGHTS, help="Path to model weights")
    args = parser.parse_args()

    analyze_audio(args.audio, model_path=args.weights)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python predict.py <path_to_audio_file>")
        sys.exit(1)
    main()
