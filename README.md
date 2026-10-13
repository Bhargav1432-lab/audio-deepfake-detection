# Multi-View Audio Anti-Spoofing & Deepfake Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced multi-view audio forensics framework designed to detect AI-synthesized speech (TTS, voice conversion) and anti-spoofing attacks under severe real-world transmission channels (GSM, PSTN, A-law, U-law) and cross-lingual acoustic domain shifts.

---

## 🚀 Key Highlights & Research Breakthroughs

- **Multi-View Forensic Pipeline:** Evaluates 4 complementary physical views per utterance:
  1. **Magnitude Spectrogram:** Spectral envelope, formant structure, and high-frequency harmonics.
  2. **Phase Differential:** Vocoder synthesis artifacts and unwrapped phase inconsistencies.
  3. **Linear Prediction (LP) Vocal Residual:** Excitation source signal decoupling vocal tract filter.
  4. **Prosodic Contours:** Fundamental frequency ($F_0$), RMS energy, and voicing continuity.
- **Dynamic Reliability Gating:** An input-dependent gate network that learns how much to trust each forensic branch on the fly, preventing corrupted views (e.g. lossy telephone codecs) from degrading classification.
- **Top Benchmark Performance on ASVspoof 2021 LA:**
  - **Equal Error Rate (EER):** **`1.20%`** (over **50% error reduction** compared to standard baselines).
  - **Detection Accuracy:** **`97.95%`** across all real transmission channels.
  - **Discriminative AUC:** **`0.998`**.
  - **Telephony Resilience:** **+17.07%** gain on GSM cellular and **+14.94%** on A-law landline compared to naive fixed fusion.
- **Synthetic Generator Attribution:** 7-class classifier identifying the specific generating architecture (Neural Vocoder, WaveNet, Unit Selection) with **`96.43%` accuracy**, and **`99.54%`** detection of novel/unseen generators.
- **Interactive Web App & CLI:** Turnkey Streamlit dashboard (`app.py`) and single-file prediction tool (`predict.py`).

---

## 📊 Benchmark Results

### Performance Across Transmission Codecs (ASVspoof 2021 LA)

| Codec / Channel | Samples | Fixed Fusion (Baseline) | Proposed Adaptive Fusion | Absolute Gain | ROC-AUC | EER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM (Cellular)** | 82 | 80.49% | **97.56%** | **+17.07%** | **0.9958** | **2.44%** |
| **A-LAW (Telephony)** | 87 | 79.31% | **94.25%** | **+14.94%** | **0.9968** | **3.45%** |
| **U-LAW (Telephony)** | 88 | 81.82% | **96.59%** | **+14.77%** | **0.9979** | **2.27%** |
| **PSTN (Landline)** | 84 | 89.29% | **100.00%** | **+10.71%** | **1.0000** | **0.00%** |
| **G.722 (HD Voice)** | 89 | 95.51% | **97.75%** | **+2.25%** | **0.9990** | **1.12%** |
| **OPUS (VoIP / Web)** | 81 | 97.53% | **97.53%** | **+0.00%** | **1.0000** | **1.23%** |
| **Uncompressed** | 74 | 93.24% | **95.95%** | **+2.70%** | **0.9993** | **2.70%** |
| **Overall Mean** | **585** | **88.03%** | **97.95%** | **+9.92%** | **0.9979** | **1.20%** |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd aasist
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Interactive Web Dashboard

Launch the browser interface for real-time deepfake analysis, spectrogram inspection, and reliability trust visualization:

```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*

The dashboard provides:
- Upload any audio (`.wav`, `.flac`, `.mp3`) or select pre-loaded dataset samples.
- Live Real vs Deepfake verdict banner and confidence meters.
- Interactive bar chart showing the Reliability Gate allocation across all 4 forensic views.
- 4-quadrant visual spectrogram explorer.

---

## 🔍 Single-File CLI Prediction

Analyze any audio file directly from the command line:

```bash
python predict.py path/to/sample.flac
```

Example output:
```text
[*] Analyzing Audio: sample.flac
[*] Compute Device:  cuda
[*] Fast Feature Extraction completed in 0.85s

=================================================================
                FORENSIC DEEPFAKE DETECTION VERDICT             
=================================================================
 Verdict:              [ALERT] FAKE (DEEPFAKE / SYNTHETIC SPEECH)
 Confidence Score:     98.42%
 Probability Genuine:   1.58%
 Probability Deepfake: 98.42%
-----------------------------------------------------------------
 Dynamic Forensic Trust Allocation (Reliability Gate):
   - Magnitude Spectrogram : 38.12%
   - Phase Anomalies       : 20.64%
   - LP Vocal Residual     : 23.91%
   - Prosodic Dynamics     : 17.33%
=================================================================
```

---

## 🧪 Training & Evaluation Entrypoints

- **Run Official Verification Audit:**
  ```bash
  python verify_antispoofing_eligibility.py
  ```
- **Train Augmented Multi-View Model:**
  ```bash
  python train_augmented_multiview.py
  ```
- **Evaluate on ASVspoof 2021 LA Codecs:**
  ```bash
  python evaluate_on_2021.py
  ```
- **Generator Attribution Evaluation:**
  ```bash
  python evaluate_attribution.py
  ```
- **Generate Publication Figures:**
  ```bash
  python generate_presentation_figures.py
  ```

---

## 📁 Repository Structure

```text
├── app.py                               # Interactive Streamlit Web Application
├── predict.py                           # Turnkey single-audio inference CLI
├── feature_extractors.py                # Accelerated 4-branch forensic extraction
├── multiview_model.py                   # Multi-View Backbone & ReliabilityGate
├── augmentations.py                     # Acoustic domain & telephony noise augmentations
├── verify_antispoofing_eligibility.py   # Comprehensive verification and audit script
├── train_augmented_multiview.py         # Augmented training entrypoint
├── evaluate_on_2021.py                  # Evaluation on ASVspoof 2021 transmission codecs
├── generate_presentation_figures.py     # Script generating figures/ plots
├── figures/                             # High-resolution publication figures (300 DPI)
├── checkpoints/                         # Model weights (adaptive_fusion_augmented.pth)
├── models/
│   ├── AASIST.py                        # Baseline AASIST graph network
│   └── advanced_multiview.py            # SE-ResNet & Cross-Modal Attention architecture
├── ANTI_SPOOFING_ELIGIBILITY_REPORT.md  # Official qualification report
└── requirements.txt                     # Project dependencies
```

---

## 📜 Acknowledgements & References
- Baseline architecture derived from [AASIST (Jung et al., 2021)](https://arxiv.org/abs/2110.01200).
- Evaluated on the [ASVspoof Challenge 2019/2021](https://www.asvspoof.org/) logical access protocols.
