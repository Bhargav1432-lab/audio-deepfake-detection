"""
app.py

Interactive Web Dashboard for Audio Anti-Spoofing & Deepfake Detection.
Built with Streamlit and PyTorch.

Features:
  - Live Audio Upload (.wav, .flac, .mp3) or Preloaded Test Samples
  - Real-Time Multi-View Feature Extraction (Magnitude, Phase, Residual, Prosody)
  - Deepfake Detection Verdict & Confidence Scores
  - Dynamic Reliability Gate Trust Visualizer
  - Multi-View Forensic Spectrograms Viewer
  - Benchmark Performance Dashboard & Publication Figures

Run using:
    streamlit run app.py
"""

import os
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

from feature_extractors import extract_all_features
from multiview_model import AdaptiveFusionModel, FixedFusionModel

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Multi-View Audio Deepfake Forensics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-box-real {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .verdict-box-fake {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .verdict-sub {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED MODEL LOADER -----------------
@st.cache_resource
def load_detector_model(model_type, device_str):
    device = torch.device(device_str)
    if model_type == "Augmented Adaptive Model (Best: 1.20% EER)":
        model = AdaptiveFusionModel().to(device)
        path = "checkpoints/adaptive_fusion_augmented.pth"
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        elif os.path.exists("adaptive_fusion.pth"):
            model.load_state_dict(torch.load("adaptive_fusion.pth", map_location=device))
    elif model_type == "Standard Adaptive Fusion Model":
        model = AdaptiveFusionModel().to(device)
        if os.path.exists("adaptive_fusion.pth"):
            model.load_state_dict(torch.load("adaptive_fusion.pth", map_location=device))
    else:  # Fixed Fusion
        model = FixedFusionModel().to(device)
        if os.path.exists("fixed_fusion.pth"):
            model.load_state_dict(torch.load("fixed_fusion.pth", map_location=device))

    model.eval()
    return model, device


# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/voice-recognition-scan.png", width=64)
st.sidebar.title("Forensic Controls")

device_option = "cuda" if torch.cuda.is_available() else "cpu"
selected_device = st.sidebar.selectbox("Compute Engine", [device_option, "cpu"] if device_option == "cuda" else ["cpu"])

model_choice = st.sidebar.selectbox(
    "Forensic Architecture",
    [
        "Augmented Adaptive Model (Best: 1.20% EER)",
        "Standard Adaptive Fusion Model",
        "Fixed Fusion Baseline (Concat)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Verified Benchmarks")
st.sidebar.markdown("""
- **Best ASVspoof 2021 EER:** **`1.20%`**
- **Codec Accuracy:** **`97.95%`**
- **Discriminative AUC:** **`0.998`**
- **Generator Attribution:** **`96.43%`**
""")
st.sidebar.markdown("---")
st.sidebar.caption("Audio Anti-Spoofing & Deepfake Detection System")


# ----------------- HEADER -----------------
st.markdown('<div class="main-header">🎙️ Multi-View Audio Deepfake Forensics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-Time Anti-Spoofing Countermeasure with Dynamic Reliability Gating across Transmission Codecs</div>', unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["🔍 Live Deepfake Analysis", "📊 Research Benchmarks & Figures", "📜 Official Eligibility Audit"])

# ================= TAB 1: LIVE ANALYSIS =================
with tab1:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        input_method = st.radio("Select Audio Source:", ["Upload Audio File", "Use Pre-loaded Benchmark Sample"], horizontal=True)

        audio_path_to_process = None

        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader("Upload an audio file (.wav, .flac, .mp3):", type=["wav", "flac", "mp3", "ogg"])
            if uploaded_file is not None:
                # Save to temporary file
                suffix = os.path.splitext(uploaded_file.name)[1]
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.read())
                tfile.flush()
                audio_path_to_process = tfile.name
                st.audio(audio_path_to_process)
        else:
            # Pre-loaded benchmark samples
            sample_options = {
                "Sample 1: LA_E_5656373 (Cellular A-law Channel)": "LA2021/flac/LA_E_5656373.flac",
                "Sample 2: LA_E_1000048 (Benchmark FLAC)": "LA2021/flac/LA_E_1000048.flac",
                "Sample 3: LA_E_1000166 (Benchmark FLAC)": "LA2021/flac/LA_E_1000166.flac",
                "Sample 4: LA_E_1000200 (Benchmark FLAC)": "LA2021/flac/LA_E_1000200.flac",
            }
            available_samples = {k: v for k, v in sample_options.items() if os.path.exists(v)}
            if available_samples:
                chosen_sample = st.selectbox("Choose sample audio from dataset:", list(available_samples.keys()))
                audio_path_to_process = available_samples[chosen_sample]
                st.audio(audio_path_to_process)
            else:
                st.warning("Sample FLAC files not found in root. Please use the upload option.")

    with col_meta:
        st.info("""
        **Forensic Detection Pipeline:**
        1. **Magnitude:** Harmonic & spectral envelope
        2. **Phase:** Vocoder phase distortion
        3. **LP Residual:** Vocal excitation anomalies
        4. **Prosody:** Pitch (F0) & energy dynamics
        """)

    if audio_path_to_process:
        if st.button("🚀 Run Forensic Deepfake Analysis", type="primary", use_container_width=True):
            with st.spinner("Extracting multi-view representations and evaluating reliability gate..."):
                t0 = time.time()
                # 1. Feature extraction
                feats = extract_all_features(audio_path_to_process, fast_prosody=True)
                extract_time = time.time() - t0

                # 2. Format tensor
                batch = {
                    "magnitude": torch.tensor(feats["magnitude"]).unsqueeze(0).to(selected_device),
                    "phase": torch.tensor(feats["phase"]).unsqueeze(0).to(selected_device),
                    "residual": torch.tensor(feats["residual"]).unsqueeze(0).to(selected_device),
                    "prosody": torch.tensor(feats["prosody"]).unsqueeze(0).to(selected_device),
                }

                # 3. Model inference
                model, dev = load_detector_model(model_choice, selected_device)
                with torch.no_grad():
                    if "Adaptive" in model_choice:
                        logits, weights = model(batch, return_weights=True)
                        w = weights[0].cpu().numpy()
                    else:
                        logits = model(batch)
                        w = np.array([0.25, 0.25, 0.25, 0.25])

                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                    prob_spoof = probs[0] * 100.0
                    prob_real = probs[1] * 100.0
                    pred_label = logits.argmax(dim=1).item()

            # ----------------- DISPLAY RESULTS -----------------
            st.markdown("---")

            if pred_label == 1:
                # Real
                st.markdown(f"""
                <div class="verdict-box-real">
                    <div class="verdict-title" style="color: #065F46;">🟢 AUTHENTIC HUMAN VOICE (BONA FIDE)</div>
                    <div class="verdict-sub" style="color: #047857;">Confidence Score: <b>{prob_real:.2f}%</b> (Genuine Human Speech Detected)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fake
                st.markdown(f"""
                <div class="verdict-box-fake">
                    <div class="verdict-title" style="color: #991B1B;">🔴 AI DEEPFAKE DETECTED (SPOOFED)</div>
                    <div class="verdict-sub" style="color: #B91C1C;">Confidence Score: <b>{prob_spoof:.2f}%</b> (Synthetic Speech / Voice Conversion Attack)</div>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Authentic (Bona Fide) Probability", f"{prob_real:.2f}%")
            m2.metric("Synthetic (Deepfake) Probability", f"{prob_spoof:.2f}%")
            m3.metric("Extraction + Analysis Latency", f"{extract_time:.2f} s")

            # ----------------- RELIABILITY GATE BREAKDOWN -----------------
            st.markdown("### ⚖️ Learned Reliability Gate Allocation")
            st.write("The network dynamically evaluates channel degradation and computes how much to trust each forensic branch for this clip:")

            gate_df = pd.DataFrame({
                "Forensic Branch": ["Magnitude Spectrogram", "Phase Anomalies", "LP Residual (Excitation)", "Prosody Dynamics"],
                "Trust Allocation (%)": [w[0] * 100, w[1] * 100, w[2] * 100, w[3] * 100]
            })

            col_chart, col_desc = st.columns([3, 2])
            with col_chart:
                st.bar_chart(gate_df.set_index("Forensic Branch"), color="#2563EB")

            with col_desc:
                st.markdown("#### Branch Summary:")
                for i, row in gate_df.iterrows():
                    st.write(f"- **{row['Forensic Branch']}:** `{row['Trust Allocation (%)']:.2f}%`")
                most_trusted = gate_df.sort_values(by="Trust Allocation (%)", ascending=False).iloc[0]
                st.success(f"**Primary Forensic Anchor:** The gate leaned most heavily on **{most_trusted['Forensic Branch']}** ({most_trusted['Trust Allocation (%)']:.1f}%).")

            # ----------------- MULTI-VIEW SPECTROGRAM VISUALIZER -----------------
            with st.expander("🔬 View Extracted Multi-View Spectrograms", expanded=False):
                fig, axes = plt.subplots(2, 2, figsize=(12, 6))

                # Magnitude
                im0 = axes[0, 0].imshow(feats["magnitude"], aspect="auto", origin="lower", cmap="magma")
                axes[0, 0].set_title("1. Magnitude Spectrogram (dB)")
                fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

                # Phase Difference
                im1 = axes[0, 1].imshow(feats["phase"], aspect="auto", origin="lower", cmap="twilight")
                axes[0, 1].set_title("2. Phase Differential Spectrogram")
                fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

                # Residual
                im2 = axes[1, 0].imshow(feats["residual"], aspect="auto", origin="lower", cmap="viridis")
                axes[1, 0].set_title("3. Linear Prediction Residual Spectrogram")
                fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

                # Prosody
                axes[1, 1].plot(feats["prosody"][0], label="F0 (Pitch Hz)", color="#e74c3c", linewidth=2)
                axes[1, 1].plot(feats["prosody"][1] * 500, label="RMS Energy (Scaled)", color="#2ecc71", linewidth=1.5)
                axes[1, 1].set_title("4. Prosodic Pitch & Energy Dynamics")
                axes[1, 1].legend(loc="upper right")
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)


# ================= TAB 2: BENCHMARK FIGURES =================
with tab2:
    st.markdown("### 📈 Experimental Results & Publication Figures")
    st.write("Consolidated charts evaluating multi-view adaptive fusion across real telephone codecs and degradation conditions:")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if os.path.exists("figures/fig1_codec_comparison.png"):
            st.image("figures/fig1_codec_comparison.png", caption="Figure 1: Anti-Spoofing Performance Across Real Transmission Codecs (ASVspoof 2021)")
        if os.path.exists("figures/fig3_gate_vs_snr_trajectory.png"):
            st.image("figures/fig3_gate_vs_snr_trajectory.png", caption="Figure 3: Reliability Gate Shift Under SNR Degradation")

    with col_f2:
        if os.path.exists("figures/fig2_reliability_weights_by_codec.png"):
            st.image("figures/fig2_reliability_weights_by_codec.png", caption="Figure 2: Reliability Gate Weights Distribution across Codecs")
        if os.path.exists("figures/fig4_attribution_confusion_matrix.png"):
            st.image("figures/fig4_attribution_confusion_matrix.png", caption="Figure 4: 7-Class Generator Attribution Confusion Matrix (96.43% Dev Acc)")

    if os.path.exists("figures/fig5_ablation_variants_eer.png"):
        st.markdown("---")
        st.image("figures/fig5_ablation_variants_eer.png", caption="Figure 5: Architectural Ablation Study (EER % per Feature Branch Isolation)")


# ================= TAB 3: AUDIT REPORT =================
with tab3:
    st.markdown("### 📋 Official Anti-Spoofing Eligibility Report")
    report_file = "ANTI_SPOOFING_ELIGIBILITY_REPORT.md"
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            report_text = f.read()
        st.markdown(report_text)
    else:
        st.info("Run `python verify_antispoofing_eligibility.py` to regenerate the official eligibility audit report.")
