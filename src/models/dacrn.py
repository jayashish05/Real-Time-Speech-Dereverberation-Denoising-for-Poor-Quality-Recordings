"""DA-CRN: degradation-aware convolutional recurrent network (proposed model) and its ablations.

Backbone: causal CRN (Tan & Wang 2018) predicting a tanh-bounded complex ratio mask (DCCRN-E style).
Contribution: a causal degradation estimator predicts per-frame [SNR, T60, C50] from the encoder
bottleneck. The estimates are trained with an auxiliary loss and drive FiLM modulation of the
bottleneck before the enhancement GRU, so processing adapts to the estimated acoustic condition.

cond_mode:
  "none"      plain CRN backbone (no estimator)
  "multitask" estimator + auxiliary loss, no FiLM   (isolates the regularisation effect)
  "film"      estimator + auxiliary loss + FiLM      (proposed)
  "oracle"    FiLM driven by ground-truth labels     (upper bound, needs labels at inference)

Every layer is causal in time: encoder convs have time kernel 2 (1 frame of history), the decoder
has time kernel 1, GRUs are unidirectional. forward() on a whole utterance equals step-wise
streaming with the caches handled in `stream_step` (see tests/test_streaming.py).
"""
import torch
import torch.nn as nn

from src.preprocessing.dsp import stft, istft

# label normalisation: (value - offset) / scale ; clean / anechoic labels are clipped to the bounds
LABELS = ("snr_db", "t60_s", "c50_db")
LABEL_RANGE = {"snr_db": (-10.0, 40.0), "t60_s": (0.0, 1.5), "c50_db": (-5.0, 40.0)}
LABEL_NORM = {"snr_db": (15.0, 20.0), "t60_s": (0.5, 0.5), "c50_db": (15.0, 20.0)}


def normalise_labels(d):
    out = []
    for k in LABELS:
        lo, hi = LABEL_RANGE[k]
        off, sc = LABEL_NORM[k]
        out.append((min(max(d[k], lo), hi) - off) / sc)
    return out


def denormalise(v):
    """[..., 3] normalised -> dict of physical values."""
    return {k: v[..., i] * LABEL_NORM[k][1] + LABEL_NORM[k][0] for i, k in enumerate(LABELS)}


class CausalConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (2, 5), stride=(1, 2), padding=(0, 2))
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x, cache=None):  # x [B, C, T, F]
        x = torch.cat([torch.zeros_like(x[:, :, :1]) if cache is None else cache, x], dim=2)
        return self.act(self.norm(self.conv(x))), x[:, :, -1:]


class Up(nn.Module):
    def __init__(self, cin, cout, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin * 2, cout, (1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1))
        self.post = nn.Identity() if last else nn.Sequential(nn.BatchNorm2d(cout), nn.PReLU(cout))

    def forward(self, x, skip):
        return self.post(self.conv(torch.cat([x, skip], dim=1)))


class DACRN(nn.Module):
    def __init__(self, channels=(16, 32, 48, 64, 64), hidden=256, est_hidden=64, cond_mode="film"):
        super().__init__()
        assert cond_mode in ("none", "multitask", "film", "oracle")
        self.cond_mode = cond_mode
        ch = (3,) + tuple(channels)
        self.enc = nn.ModuleList(CausalConv(ch[i], ch[i + 1]) for i in range(len(channels)))
        self.dec = nn.ModuleList(Up(ch[i + 1], ch[i] if i else 2, last=i == 0) for i in reversed(range(len(channels))))
        nf = 256 // 2 ** len(channels)
        d = channels[-1] * nf
        self.gru = nn.GRU(d, hidden, num_layers=2, batch_first=True)
        self.proj = nn.Linear(hidden, d)
        if cond_mode != "none":
            self.est_gru = nn.GRU(d, est_hidden, batch_first=True)
            self.est_out = nn.Linear(est_hidden, len(LABELS))
        if cond_mode in ("film", "oracle"):
            self.film = nn.Sequential(nn.Linear(len(LABELS), 64), nn.PReLU(), nn.Linear(64, 2 * d))
            nn.init.zeros_(self.film[-1].weight)  # start as identity modulation
            nn.init.zeros_(self.film[-1].bias)

    # ---- core on a block of frames (whole utterance or a single streaming frame)
    def _core(self, X, state, cond=None):
        """X complex [B, F=257, T]; state dict of caches. Returns (mask [B,257,T] complex, est, state)."""
        mag = X.abs().clamp_min(1e-8)
        comp = mag ** 0.3
        feat = torch.stack([comp * X.real / mag, comp * X.imag / mag, comp], dim=1)[..., :256, :]
        h = feat.transpose(2, 3)  # [B, 3, T, F]
        skips, new = [], {}
        for i, layer in enumerate(self.enc):
            h, new[f"c{i}"] = layer(h, state.get(f"c{i}"))
            skips.append(h)
        B, C, T, Fb = h.shape
        z = h.permute(0, 2, 1, 3).reshape(B, T, C * Fb)
        est = None
        if self.cond_mode != "none":
            e, new["est"] = self.est_gru(z, state.get("est"))
            est = self.est_out(e)
        if self.cond_mode in ("film", "oracle"):
            c = est if self.cond_mode == "film" else cond
            g, b = self.film(c).chunk(2, dim=-1)
            z = z * (1 + g) + b
        r, new["gru"] = self.gru(z, state.get("gru"))
        h = self.proj(r).reshape(B, T, C, Fb).permute(0, 2, 1, 3)
        for layer, s in zip(self.dec, reversed(skips)):
            h = layer(h, s)
        m = h.transpose(2, 3)  # [B, 2, 256, T]
        m = torch.cat([m, m[:, :, -1:]], dim=2)  # Nyquist bin reuses the last mask
        mc = torch.complex(m[:, 0], m[:, 1])
        mask = torch.tanh(mc.abs()) * torch.exp(1j * torch.angle(mc))
        return mask, est, new

    def forward(self, wav, cond=None):
        """wav [B, N] -> (enhanced [B, N], estimates [B, T, 3] or None)."""
        X = stft(wav)
        if cond is not None and cond.dim() == 2:
            cond = cond[:, None, :].expand(-1, X.shape[-1], -1)
        mask, est, _ = self._core(X, {}, cond)
        return istft(X * mask, wav.shape[-1]), est

    @torch.no_grad()
    def stream_step(self, X_frame, state, cond=None):
        """One STFT frame [B, 257, 1] -> enhanced frame, estimate, new state (all caches)."""
        mask, est, state = self._core(X_frame, state, cond)
        return X_frame * mask, est, state


if __name__ == "__main__":
    for mode in ("none", "film"):
        m = DACRN(cond_mode=mode)
        y, est = m(torch.randn(2, 16000))
        print(mode, sum(p.numel() for p in m.parameters()), tuple(y.shape), None if est is None else tuple(est.shape))
