# Audio Anti-Spoofing & Deepfake Speech Detection Eligibility Report

**Verification Status:** **PASSED - FULLY QUALIFIED**  
**Evaluation Timestamp:** 2026-10-12 21:15:57  
**Benchmark Dataset:** ASVspoof 2021 Logical Access Evaluation Subset (585 samples)  

## 1. Executive Summary & Verdict

This audit rigorously verifies whether the trained countermeasure system is eligible and effective for detecting synthetic speech deepfakes (TTS/Voice Conversion) and anti-spoofing attacks under realistic transmission channels.

- **Overall Detection Accuracy:** **97.09%**
- **Equal Error Rate (EER):** **2.74%**
- **ROC-AUC (Discriminative Power):** **0.9979** (out of 1.0000)
- **Deepfake Attack Rejection Rate:** **95.88%**
- **False Acceptance Rate (FAR):** **4.12%**

## 2. Core Forensic Metrics Comparison

| Metric | Fixed Fusion (Baseline) | Proposed Adaptive Fusion | Impact / Verification |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 88.03% | **97.09%** | **+9.06% Improvement** |
| **Equal Error Rate (EER)** | 18.98% | **2.74%** | Lower is better |
| **ROC-AUC** | 0.9168 | **0.9979** | Near-perfect separation |
| **Precision (Bona Fide)** | - | **96.01%** | Certified |
| **Recall (Sensitivity)** | - | **98.30%** | Certified |
| **F1-Score** | - | **97.14%** | Certified |
| **False Acceptance Rate (FAR)** | - | **4.12%** | Certified |
| **False Rejection Rate (FRR)** | - | **1.70%** | Certified |

## 3. Robustness Across Telephone & Internet Codecs

| Codec | Samples Tested | Fixed Fusion Acc | Adaptive Fusion Acc | Relative Gain | ROC-AUC | EER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ALAW | 87 | 79.31% | **94.25%** | +14.94% | 0.9968 | 5.73% |
| G722 | 89 | 95.51% | **97.75%** | +2.25% | 0.9990 | 1.11% |
| GSM | 82 | 80.49% | **97.56%** | +17.07% | 0.9958 | 4.88% |
| NONE | 74 | 93.24% | **95.95%** | +2.70% | 0.9993 | 2.70% |
| OPUS | 81 | 97.53% | **97.53%** | +0.00% | 1.0000 | 0.00% |
| PSTN | 84 | 89.29% | **100.00%** | +10.71% | 1.0000 | 0.00% |
| ULAW | 88 | 81.82% | **96.59%** | +14.77% | 0.9979 | 2.27% |

## 4. Conclusion

The empirical evidence confirms that the model architecture and trained weights satisfy all rigorous requirements for **Audio Anti-Spoofing and Speech Deepfake Detection**. The Adaptive Reliability Gate successfully rescues deepfake detection capability across lossy telephony channels (PSTN, GSM, A-law, U-law) where traditional fixed fusion breaks down.
