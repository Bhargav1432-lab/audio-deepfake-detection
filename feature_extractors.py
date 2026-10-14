"""
feature_extractors.py

Unified module containing all 4 feature extraction functions, validated
individually earlier. Import this in any script that needs to compute
magnitude, phase, residual, or prosody features from raw audio.

Each function takes a raw audio waveform (1D numpy array) and returns a
FIXED-SIZE numpy array, so it can be batched by PyTorch's DataLoader.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import scipy.signal
import torch

# ---------------- CONFIG ----------------
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
LPC_ORDER = 16

# Fixed sizes -- audio will be padded/trimmed so all outputs have these
# exact shapes, which PyTorch requires for batching.
TARGET_AUDIO_LEN = 64600          # same as AASIST's NB_SAMP
N_TIME_FRAMES = 126               # ~ 64600 / 512, fixed via padding below
N_FREQ_BINS = N_FFT // 2 + 1      # 1025
# -----------------------------------------


def pad_or_trim_audio(audio, target_len=TARGET_AUDIO_LEN):
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(target_len / len(audio)) + 1
    return np.tile(audio, num_repeats)[:target_len]


def pad_or_trim_2d(feat, target_frames=N_TIME_FRAMES):
    """Pad/trim the TIME axis (axis=1) of a 2D feature (freq_bins x time)."""
    current_frames = feat.shape[1]
    if current_frames >= target_frames:
        return feat[:, :target_frames]
    pad_width = target_frames - current_frames
    return np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")


def pad_or_trim_1d(feat, target_len):
    if len(feat) >= target_len:
        return feat[:target_len]
    return np.pad(feat, (0, target_len - len(feat)), mode="constant")


def extract_magnitude(audio):
    stft_result = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft_result)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    magnitude_db = pad_or_trim_2d(magnitude_db)
    return magnitude_db.astype(np.float32)  # shape: (1025, 126)


def extract_phase(audio):
    stft_result = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.angle(stft_result)
    phase_unwrapped = np.unwrap(phase, axis=1)
    phase_diff = np.diff(phase_unwrapped, axis=1)
    phase_diff = pad_or_trim_2d(phase_diff, target_frames=N_TIME_FRAMES - 1)
    return phase_diff.astype(np.float32)  # shape: (1025, 125)


def extract_residual(audio):
    frame_length = 400
    hop_length = 160
    residual = np.zeros_like(audio)
    num_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        if len(frame) < LPC_ORDER + 1:
            continue
        try:
            lpc_coeffs = librosa.lpc(frame, order=LPC_ORDER)
            predicted = np.zeros_like(frame)
            for n in range(LPC_ORDER, len(frame)):
                predicted[n] = -np.sum(lpc_coeffs[1:] * frame[n - LPC_ORDER:n][::-1])
            residual[start:end] += (frame - predicted)
        except Exception:
            continue

    # convert residual (1D waveform) into a spectrogram for CNN input,
    # same shape convention as magnitude/phase for architectural consistency
    stft_result = librosa.stft(residual, n_fft=N_FFT, hop_length=HOP_LENGTH)
    residual_mag = np.abs(stft_result)
    residual_db = librosa.amplitude_to_db(residual_mag, ref=np.max(residual_mag) + 1e-8)
    residual_db = pad_or_trim_2d(residual_db)
    return residual_db.astype(np.float32)  # shape: (1025, 126)


def extract_prosody(audio, fast=False):
    """
    Extracts F0, RMS energy, and Voiced probability using librosa.pyin,
    matching the exact distribution the model weights were trained on.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio, sr=SR,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0_clean = np.nan_to_num(f0, nan=0.0)
    voiced_flag_numeric = (voiced_flag.astype(np.float32) if voiced_flag is not None 
                           else (f0_clean > 0.0).astype(np.float32))
    rms = librosa.feature.rms(y=audio)[0]

    # fixed-length 1D vectors
    target_len = N_TIME_FRAMES
    f0_clean = pad_or_trim_1d(f0_clean, target_len)
    rms = pad_or_trim_1d(rms, target_len)
    voiced_flag_numeric = pad_or_trim_1d(voiced_flag_numeric, target_len)

    # stack into a single feature vector: (3, N_TIME_FRAMES)
    prosody_features = np.stack([f0_clean, rms, voiced_flag_numeric], axis=0)
    return prosody_features.astype(np.float32)


def extract_all_features(filepath, fast_prosody=False):
    """
    Loads audio and extracts all 4 branches for one file.
    Returns a dict of numpy arrays, ready to be converted to tensors.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(filepath, sr=SR)
    
    audio = pad_or_trim_audio(audio)

    return {
        "magnitude": extract_magnitude(audio),
        "phase": extract_phase(audio),
        "residual": extract_residual(audio),
        "prosody": extract_prosody(audio, fast=fast_prosody),
    }
