"""Classical baselines (no training):
  wiener      decision-directed Wiener filter (Ephraim-Malah a-priori SNR) with minimum-statistics noise PSD
  wpe         single-channel Weighted Prediction Error dereverberation (Nakatani 2010, via nara_wpe)
  wpe_wiener  cascade: WPE (late reverb) then Wiener (noise) -- the classical joint system
All operate on 16 kHz mono float arrays and use the project STFT.
"""
import numpy as np
import torch
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from src.preprocessing.dsp import istft, stft


def _stft(x):
    return stft(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy().astype(np.complex128)


def _istft(X, n):
    return istft(torch.from_numpy(X.astype(np.complex64))[None], n)[0].numpy()


def wiener(x, alpha_dd=0.98, win_s=1.5, bias=1.5, g_min=0.1):
    X = _stft(x)  # [F, T]
    P = np.abs(X) ** 2
    Ps = uniform_filter1d(P, 5, axis=1)  # smoothed periodogram
    D = max(int(win_s * 16000 / 256), 3)
    noise = bias * minimum_filter1d(Ps, D, axis=1, origin=-(D // 2))  # trailing (causal) minimum
    noise = np.maximum(noise, 1e-12)
    G = np.ones_like(P)
    prev = np.zeros(P.shape[0])
    for t in range(P.shape[1]):
        gamma = P[:, t] / noise[:, t]
        xi = alpha_dd * prev / noise[:, t] + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
        G[:, t] = np.maximum(xi / (1 + xi), g_min)
        prev = (G[:, t] ** 2) * P[:, t]
    return _istft(X * G, len(x)).astype(np.float32)


def wpe(x, taps=10, delay=3, iterations=3):
    from nara_wpe.wpe import wpe as nara_wpe
    X = _stft(x)
    Y = nara_wpe(X[:, None, :], taps=taps, delay=delay, iterations=iterations)[:, 0, :]
    return _istft(Y, len(x)).astype(np.float32)


def wpe_wiener(x):
    return wiener(wpe(x))


CLASSICAL = {"wiener": wiener, "wpe": wpe, "wpe_wiener": wpe_wiener}

if __name__ == "__main__":
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(0)
    t = np.arange(48000) / 16000
    s = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    h = rng.standard_normal(8000) * np.exp(-6.9 * np.arange(8000) / 16000 / 0.6); h[0] = 5
    rev = fftconvolve(s, h)[:48000].astype(np.float32)
    noisy = s + 0.3 * rng.standard_normal(48000).astype(np.float32)
    err = lambda y: 10 * np.log10(np.sum(s ** 2) / np.sum((y / (np.dot(y, s) / np.dot(s, s)) - s) ** 2))
    print("wiener  in %.1f dB -> out %.1f dB" % (err(noisy), err(wiener(noisy))))
    print("wpe     in %.1f dB -> out %.1f dB" % (err(rev), err(wpe(rev))))
