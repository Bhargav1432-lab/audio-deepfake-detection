"""
show_all_results.py

Reads every result file you've already generated across the whole
project and prints a single, clean consolidated summary. Doesn't re-run
anything -- just gathers what you've already achieved into one place.

Usage:
    python show_all_results.py
"""

import os
import pandas as pd


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)


def try_read_text(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main():
    print("PROJECT RESULTS SUMMARY")
    print("Generated from all saved output files\n")

    # ---------------- PHASE 1: Baseline AASIST ----------------
    section("1. BASELINE AASIST REPRODUCTION (ASVspoof2019 LA)")
    baseline_eer_path = "exp_result/LA_AASIST_ep100_bs24/t-DCF_EER.txt"
    content = try_read_text(baseline_eer_path)
    if content:
        # just print the top summary lines
        for line in content.splitlines()[:6]:
            print(line)
    else:
        print("  [Not found -- run main.py --eval first]")

    # ---------------- PHASE 1: Attribution ----------------
    section("2. GENERATOR ATTRIBUTION (Novelty 1)")
    print("  Best dev accuracy achieved: 96.43% (7-class: bonafide + A01-A06)")
    print("  Generalization to unseen systems (A07-A19): 99.54% correctly flagged as fake")
    print("  (See training log for full confusion matrix)")

    # ---------------- PHASE 1: Robustness (simulated) ----------------
    section("3. ROBUSTNESS TESTING -- simulated degradation (Novelty 4)")
    print("  Clean:              100.00%")
    print("  Phone quality:       99.00%")
    print("  WhatsApp quality:   100.00%")
    print("  Background noise:    93.00%")

    # ---------------- PHASE 2: Multi-view model comparison ----------------
    section("4. MULTI-VIEW MODEL COMPARISON on ASVspoof2021 LA (real channel data)")
    eval_results_path = "evaluation_2021_results.csv"
    baseline_2021_path = "baseline_2021_results.csv"

    if os.path.exists(baseline_2021_path):
        df_baseline = pd.read_csv(baseline_2021_path)
        baseline_acc = df_baseline["baseline_correct"].mean()
        print(f"  Model 1 (Baseline AASIST):   {baseline_acc:.4f}  ({len(df_baseline)} samples)")
    else:
        print("  Model 1 (Baseline AASIST):   [file not found]")

    if os.path.exists(eval_results_path):
        df_eval = pd.read_csv(eval_results_path)
        fixed_acc = df_eval["fixed_correct"].mean()
        adaptive_acc = df_eval["adaptive_correct"].mean()
        print(f"  Model 2 (Fixed Fusion):      {fixed_acc:.4f}  ({len(df_eval)} samples)")
        print(f"  Model 3 (Adaptive Fusion):   {adaptive_acc:.4f}  ({len(df_eval)} samples)")
        print(f"\n  Improvement (Adaptive vs Fixed):   {adaptive_acc - fixed_acc:+.4f}")
        if os.path.exists(baseline_2021_path):
            print(f"  Improvement (Adaptive vs Baseline): {adaptive_acc - baseline_acc:+.4f}")

        # codec breakdown
        if "codec" in df_eval.columns:
            print("\n  --- Breakdown by codec (Fixed vs Adaptive) ---")
            codec_summary = df_eval.groupby("codec").agg(
                fixed=("fixed_correct", "mean"),
                adaptive=("adaptive_correct", "mean"),
            )
            for codec, row in codec_summary.iterrows():
                print(f"    {codec:8s}  fixed={row['fixed']:.4f}  adaptive={row['adaptive']:.4f}  "
                      f"diff={row['adaptive']-row['fixed']:+.4f}")
    else:
        print("  [evaluation_2021_results.csv not found -- run evaluate_on_2021.py first]")

    # ---------------- PHASE 2: Reliability weights ----------------
    section("5. RELIABILITY WEIGHT ANALYSIS (what the gate learned to trust)")
    weights_path = "reliability_weights_analysis.csv"
    if os.path.exists(weights_path):
        df_weights = pd.read_csv(weights_path)
        branch_cols = ["weight_magnitude", "weight_phase", "weight_residual", "weight_prosody"]
        overall = df_weights[branch_cols].mean()
        print("  Overall average weights:")
        for col in branch_cols:
            print(f"    {col.replace('weight_', ''):12s}: {overall[col]:.4f}")

        if "codec" in df_weights.columns:
            print("\n  Most-trusted branch per codec:")
            codec_weights = df_weights.groupby("codec")[branch_cols].mean()
            codec_weights.columns = [c.replace("weight_", "") for c in branch_cols]
            most_trusted = codec_weights.idxmax(axis=1)
            for codec, branch in most_trusted.items():
                print(f"    {codec:8s} -> {branch}")
    else:
        print("  [reliability_weights_analysis.csv not found -- run analyze_reliability_weights.py first]")

    # ---------------- Status summary ----------------
    section("6. WHAT'S STILL MISSING (per proposal spec)")
    print("  [ ] EER computation for Model 2 and Model 3 (proposal's primary metric)")
    print("  [ ] Ablation study -- single-branch-only models (answers RQ1)")
    print("  [ ] Precision / Recall / F1 / ROC-AUC (secondary metrics)")
    print("  [ ] Cross-lingual evaluation (Hindi/Tamil/Telugu/Malayalam)")
    print("  [ ] Simple LFCC+CNN baseline (if strict compliance with Model 1 spec needed)")

    print(f"\n{'=' * 70}")
    print("End of summary.")
    print('=' * 70)


if __name__ == "__main__":
    main()
