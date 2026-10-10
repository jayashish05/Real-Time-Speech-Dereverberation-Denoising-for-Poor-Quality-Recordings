"""Signal processing shared by training, offline inference and streaming.

STFT convention (identical offline and streaming, so a streaming run reproduces the offline output):
frame t covers samples [t*hop - (n_fft-hop), t*hop + hop), sqrt-Hann analysis and synthesis windows,
50 % overlap (sqrt-Hann^2 = Hann sums to 1). Algorithmic latency = n_fft samples (32 ms at 16 kHz).
"""
import math

import numpy as np
import torch
from scipy.signal import fftconvolve, resample_poly

SR = 16000
N_FFT = 512
HOP = 256


def window(n_fft=N_FFT, device=None):
    return torch.hann_window(n_fft, periodic=True, device=device).sqrt()


def stft(x, n_fft=N_FFT, hop=HOP):
    """x: [B, N] -> complex [B, F, T] with T = ceil(N / hop) + 1."""
    n = x.shape[-1]
    pad_r = (-n) % hop + hop
    x = torch.nn.functional.pad(x, (n_fft - hop, pad_r))
    frames = x.unfold(-1, n_fft, hop) * window(n_fft, x.device)
    return torch.fft.rfft(frames, dim=-1).transpose(-1, -2)


def istft(X, length, n_fft=N_FFT, hop=HOP):
    """Inverse of stft(): complex [B, F, T] -> [B, length]."""
    frames = torch.fft.irfft(X.transpose(-1, -2), n=n_fft, dim=-1) * window(n_fft, X.device)
    B, T, _ = frames.shape
    y = torch.nn.functional.fold(frames.transpose(1, 2), output_size=(1, (T - 1) * hop + n_fft),
                                 kernel_size=(1, n_fft), stride=(1, hop)).reshape(B, -1)
    return y[:, n_fft - hop: n_fft - hop + length]


# ---------------------------------------------------------------- audio utils

def to_mono_16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def set_level(x, target_db, *others):
    g = 10 ** ((target_db - rms_db(x)) / 20)
    return (x * g, *(o * g for o in others))


# ---------------------------------------------------------------- RIR analysis

def direct_path_index(h):
    return int(np.argmax(np.abs(h)))


def rir_params(h, sr=SR):
    """T60 (s, Schroeder EDC, T20 extrapolation), C50 (dB), DRR (dB) of a RIR."""
    d = direct_path_index(h)
    h = h[d:]
    e = h ** 2
    # noise-floor compensation (Lundeby-style): measured RIRs end in a noise floor that would
    # otherwise dominate the energy decay curve -> cut where the envelope meets the floor, subtract it
    floor = np.mean(e[-max(len(e) // 10, 1):])
    env = np.convolve(e, np.ones(160) / 160, mode="same")  # 10 ms envelope
    below = np.nonzero(env[160:] < floor * 10 ** 0.5)[0]
    cut = 160 + below[0] if len(below) else len(e)
    e = np.maximum(e[:cut] - floor, 0) if floor > 1e-6 * e[0] else e
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-20) + 1e-20)
    # T20 fit between -5 and -25 dB, extrapolated to -60 dB
    i5, i25 = np.argmax(edc_db <= -5), np.argmax(edc_db <= -25)
    if i25 > i5 + 2:
        t = np.arange(i5, i25) / sr
        slope = np.polyfit(t, edc_db[i5:i25], 1)[0]
        t60 = float(-60 / slope) if slope < 0 else 0.0
    else:
        t60 = 0.0
    n50 = int(0.05 * sr)
    e = h[:cut] ** 2 if floor > 1e-6 * h[0] ** 2 else h ** 2
    c50 = 10 * np.log10(e[:n50].sum() / (e[n50:].sum() + 1e-20))
    nd = int(0.0025 * sr)  # +-2.5 ms direct-path window
    drr = 10 * np.log10(e[:nd].sum() / (e[nd:].sum() + 1e-20))
    return t60, float(c50), float(drr)


def prepare_rir(h, sr=SR):
    """Cut pre-delay, normalise to unit direct-path energy and return (rir, direct_path_rir).
    Reverberant speech = s * rir; the dereverberation target = s * direct_path_rir (time aligned)."""
    h = np.asarray(h, dtype=np.float32)
    if h.ndim == 2:
        h = h[:, 0]
    d = direct_path_index(h)
    pre = int(0.0025 * sr)
    h = h[max(0, d - pre):]
    d = min(d, pre)
    direct = np.zeros_like(h)
    direct[: d + pre] = h[: d + pre]
    g = 1.0 / (np.sqrt(np.sum(direct ** 2)) + 1e-12)
    return h * g, direct * g


# ---------------------------------------------------------------- degradation

def degrade(clean, rng, noise=None, rir=None, snr_db=None, clip=None, level_db=None):
    """Simulate one poor-quality recording. Returns (noisy, target, info).
    Order: clean -> room (rir) -> + noise at snr_db w.r.t. reverberant speech -> level -> clipping."""
    n = len(clean)
    if rir is not None:
        h, hd = rir
        reverb = fftconvolve(clean, h)[:n].astype(np.float32)
        target = fftconvolve(clean, hd)[:n].astype(np.float32)
    else:
        reverb, target = clean.copy(), clean.copy()
    mix = reverb.copy()
    if noise is not None:
        p_s = np.mean(reverb ** 2) + 1e-12
        p_n = np.mean(noise ** 2) + 1e-12
        mix = mix + noise * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    level_db = rng.uniform(-35, -15) if level_db is None else level_db
    mix, target = set_level(mix, level_db, target)
    if clip is not None:  # clip at a fraction of the peak (recording-chain overload)
        thr = clip * np.max(np.abs(mix))
        mix = np.clip(mix, -thr, thr)
    peak = max(np.max(np.abs(mix)), np.max(np.abs(target)))
    if peak > 0.99:
        mix, target = mix * 0.99 / peak, target * 0.99 / peak
    return mix.astype(np.float32), target.astype(np.float32)


def phone_quality(x):
    """Unseen-condition simulation: 8 kHz narrow-band telephone channel with mu-law companding."""
    from scipy.signal import butter, sosfilt
    y = resample_poly(x, 1, 2)
    y = sosfilt(butter(6, [300, 3400], btype="band", fs=8000, output="sos"), y)
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
    q = np.round(q * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return resample_poly(y, 2, 1).astype(np.float32)[: len(x)]


if __name__ == "__main__":  # self-check: perfect reconstruction and RIR parameters
    x = torch.randn(2, 16000 * 3 + 123)
    y = istft(stft(x), x.shape[-1])
    assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()
    t = np.arange(SR) / SR
    h = np.random.default_rng(0).standard_normal(SR) * np.exp(-6.9 * t / 0.5)  # T60 = 0.5 s
    h[0] = 30
    t60, c50, drr = rir_params(h)
    assert abs(t60 - 0.5) < 0.05, t60
    print("dsp ok", t60, c50, drr)
