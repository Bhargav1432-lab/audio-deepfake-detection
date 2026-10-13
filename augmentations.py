"""
augmentations.py

Acoustic Domain Shift Augmentations:
Simulates real-world mobile phone, telephony, and ambient room recording conditions
on clean studio audio (e.g., ASVspoof 2019) to prevent models from learning
spurious correlations between 'clean' and 'bonafide' speech.

Augmentations included:
1. Codec / Telephony Simulation (A-law / mu-law quantization, telephone bandpass 300-3400 Hz)
2. Additive Background Noise with random SNR (10 - 25 dB)
3. Room Impulse Response / Synthetic Reverberation (RT60 decay)
4. Dynamic Range & Frequency Response Tilt
"""

import numpy as np
import scipy.signal
import torch


def apply_telephone_bandpass(audio, sr=16000, lowcut=300.0, highcut=3400.0, order=5):
    """Simulates telephone PSTN bandwidth (300 Hz - 3400 Hz)."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, audio).astype(np.float32)


def apply_companding_codec(audio, mode='alaw'):
    """Simulates 8-bit A-law or mu-law quantization."""
    # Scale to [-1, 1]
    peak = max(np.max(np.abs(audio)), 1e-6)
    norm_audio = np.clip(audio / peak, -1.0, 1.0)

    if mode == 'mulaw':
        mu = 255.0
        companded = np.sign(norm_audio) * np.log1p(mu * np.abs(norm_audio)) / np.log1p(mu)
        # Quantize to 8-bit integers
        quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
        # Expand
        expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
    else:  # alaw
        A = 87.6
        abs_x = np.abs(norm_audio)
        companded = np.where(
            abs_x < 1.0 / A,
            (A * abs_x) / (1.0 + np.log(A)),
            (1.0 + np.log(A * abs_x)) / (1.0 + np.log(A))
        ) * np.sign(norm_audio)
        quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
        abs_q = np.abs(quantized)
        expanded = np.where(
            abs_q < 1.0 / (1.0 + np.log(A)),
            abs_q * (1.0 + np.log(A)) / A,
            np.exp(abs_q * (1.0 + np.log(A)) - 1.0) / A
        ) * np.sign(quantized)

    return (expanded * peak).astype(np.float32)


def apply_synthetic_reverb(audio, sr=16000, rt60=0.25):
    """Simulates room reverberation using an exponentially decaying white-noise impulse response."""
    duration = max(int(sr * rt60), 100)
    decay = np.exp(-3.0 * np.linspace(0, 1, duration))
    noise = np.random.randn(duration)
    rir = decay * noise
    rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-8)

    reverberated = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    # Mix wet and dry signal
    mixed = 0.7 * audio + 0.3 * reverberated
    return mixed.astype(np.float32)


def apply_additive_noise(audio, snr_db=15.0):
    """Adds white/colored noise at a specified SNR (dB)."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


class AcousticDomainAugmenter:
    """
    Randomly applies realistic channel, noise, and codec augmentations.
    """
    def __init__(self, p=0.5, sr=16000):
        self.p = p
        self.sr = sr

    def __call__(self, audio):
        if np.random.rand() > self.p:
            return audio

        out = audio.copy()
        aug_type = np.random.choice(['telephony', 'reverb', 'noise', 'codec_noise'])

        if aug_type == 'telephony':
            out = apply_telephone_bandpass(out, sr=self.sr)
            out = apply_companding_codec(out, mode='alaw')
        elif aug_type == 'reverb':
            rt60 = np.random.uniform(0.15, 0.35)
            out = apply_synthetic_reverb(out, sr=self.sr, rt60=rt60)
        elif aug_type == 'noise':
            snr = np.random.uniform(10.0, 25.0)
            out = apply_additive_noise(out, snr_db=snr)
        elif aug_type == 'codec_noise':
            snr = np.random.uniform(12.0, 22.0)
            out = apply_additive_noise(out, snr_db=snr)
            out = apply_companding_codec(out, mode='mulaw')

        # Normalize amplitude safely
        peak = np.max(np.abs(out)) + 1e-8
        if peak > 1.0:
            out = out / peak

        return out.astype(np.float32)
