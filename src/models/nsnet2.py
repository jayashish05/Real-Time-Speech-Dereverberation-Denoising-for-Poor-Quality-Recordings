"""NSNet2-style lightweight neural baseline (Braun & Tashev 2020, the DNS-challenge baseline):
log-power spectrum -> FC -> 2xGRU -> 2xFC -> sigmoid magnitude mask; the noisy phase is kept.
Re-implemented and trained on our data, with the same STFT as DA-CRN. Causal."""
import torch
import torch.nn as nn

from src.preprocessing.dsp import stft, istft


class NSNet2(nn.Module):
    cond_mode = "none"

    def __init__(self, n_bins=257, hidden=400):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_bins, hidden), nn.PReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, 600), nn.PReLU(), nn.Linear(600, 600), nn.PReLU(),
                                 nn.Linear(600, n_bins), nn.Sigmoid())

    def _core(self, X, state):
        f = torch.log10(X.abs().pow(2) + 1e-9).transpose(1, 2)  # [B, T, F]
        h, s = self.gru(self.inp(f), state.get("gru"))
        return self.out(h).transpose(1, 2), {"gru": s}

    def forward(self, wav, cond=None):
        X = stft(wav)
        mask, _ = self._core(X, {})
        return istft(X * mask, wav.shape[-1]), None

    @torch.no_grad()
    def stream_step(self, X_frame, state, cond=None):
        mask, state = self._core(X_frame, state)
        return X_frame * mask, None, state
