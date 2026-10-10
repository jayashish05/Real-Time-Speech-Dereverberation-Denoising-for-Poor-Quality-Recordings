"""Degradation simulation must produce the requested SNR, aligned targets, and correct RIR parameters."""
import numpy as np

from src.preprocessing.dsp import degrade, prepare_rir, rir_params


def test_snr_is_respected():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(32000).astype(np.float32)
    n = rng.standard_normal(32000).astype(np.float32)
    for snr in (-5, 0, 10, 20):
        mix, tgt = degrade(s, rng, noise=n, snr_db=snr, level_db=-30)
        resid = mix - tgt
        measured = 10 * np.log10(np.sum(tgt ** 2) / np.sum(resid ** 2))
        assert abs(measured - snr) < 0.05, (snr, measured)


def test_reverb_target_is_aligned_direct_path():
    rng = np.random.default_rng(1)
    t = np.arange(8000) / 16000
    h = np.concatenate([np.zeros(40), rng.standard_normal(8000) * np.exp(-6.9 * t / 0.4)]).astype(np.float32)
    h[40] = 8.0
    rir = prepare_rir(h)
    s = rng.standard_normal(16000).astype(np.float32)
    _, tgt = degrade(s, rng, rir=rir, level_db=-30)
    c = np.correlate(tgt, s, "full")
    assert abs(int(np.argmax(np.abs(c))) - (len(s) - 1)) <= 40  # pre-delay removed (only +-2.5 ms window)
    t60, c50, _ = rir_params(h)
    assert 0.3 < t60 < 0.5 and c50 > 0
