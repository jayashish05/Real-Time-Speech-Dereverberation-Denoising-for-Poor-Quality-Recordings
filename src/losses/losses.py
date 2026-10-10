"""Losses: negative SI-SDR, multi-resolution STFT (spectral convergence + log magnitude), and the
auxiliary degradation-estimation loss."""
import torch
import torch.nn.functional as F


def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR in dB, per item. est, ref: [B, N]."""
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)


def mr_stft(est, ref, resolutions=((512, 128), (1024, 256), (256, 64))):
    loss = 0.0
    for n_fft, hop in resolutions:
        w = torch.hann_window(n_fft, device=est.device)
        E = torch.stft(est, n_fft, hop, window=w, return_complex=True).abs().clamp_min(1e-7)
        R = torch.stft(ref, n_fft, hop, window=w, return_complex=True).abs().clamp_min(1e-7)
        sc = torch.linalg.norm(R - E, dim=(-2, -1)) / torch.linalg.norm(R, dim=(-2, -1)).clamp_min(1e-7)
        loss = loss + sc.mean() + F.l1_loss(E.log(), R.log())
    return loss / len(resolutions)


class EnhancementLoss:
    def __init__(self, w_sisdr=0.1, w_stft=1.0, w_aux=0.5):
        self.w_sisdr, self.w_stft, self.w_aux = w_sisdr, w_stft, w_aux

    def __call__(self, est, ref, est_labels=None, labels=None):
        parts = {"sisdr": -si_sdr(est, ref).mean()}
        total = self.w_sisdr * parts["sisdr"]
        if self.w_stft:
            parts["stft"] = mr_stft(est, ref)
            total = total + self.w_stft * parts["stft"]
        if est_labels is not None and labels is not None and self.w_aux:
            parts["aux"] = F.smooth_l1_loss(est_labels, labels[:, None, :].expand_as(est_labels).contiguous())
            total = total + self.w_aux * parts["aux"]
        return total, {k: float(v) for k, v in parts.items()}
