"""On-the-fly degradation dataset (training) and fixed-set reader (validation / test).

Each training item: random clean crop -> sampled condition (clean / noise / reverb / noise+reverb)
-> optional clipping -> random level. Labels [SNR, T60, C50] come from the simulation itself.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.models.dacrn import normalise_labels
from src.preprocessing.dsp import degrade, prepare_rir

NO_NOISE_SNR, NO_REVERB = 40.0, {"t60_s": 0.0, "c50_db": 40.0}


def read_segment(path, n, rng, tile=True):
    """Random n-sample segment; shorter files are tiled (noise) or zero-padded (speech)."""
    info = sf.info(path)
    if info.frames > n:
        start = int(rng.integers(0, info.frames - n))
        x, _ = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
    else:
        x, _ = sf.read(path, dtype="float32", always_2d=True)
        if not tile:
            return np.pad(x[:, 0], (0, n - len(x)))
        x = np.tile(x, (int(np.ceil(n / max(len(x), 1))), 1))[:n]
    return x[:, 0]


def load_rir(path):
    h, _ = sf.read(path, dtype="float32", always_2d=True)
    return prepare_rir(h[:, 0])


def simulate(clean, rng, condition, noises, rirs, snr_range=(-5, 20), clip_prob=0.0):
    """condition in {clean, noise, reverb, noise_reverb}; noises/rirs are metadata DataFrames.
    Returns noisy, target, labels dict (physical units) and the chosen parameters."""
    n = len(clean)
    info = {"condition": condition, "snr_db": NO_NOISE_SNR, **NO_REVERB, "noise": "", "rir": "", "clip": 0.0}
    noise = rir = None
    snr = None
    if "noise" in condition:
        row = noises.iloc[int(rng.integers(len(noises)))]
        noise = read_segment(row.path, n, rng)
        snr = float(rng.uniform(*snr_range))
        info.update(snr_db=snr, noise=row.path)
    if "reverb" in condition:
        row = rirs.iloc[int(rng.integers(len(rirs)))]
        rir = load_rir(row.path)
        info.update(t60_s=float(row.t60_s), c50_db=float(row.c50_db), rir=row.path)
    clip = float(rng.uniform(0.3, 0.8)) if rng.random() < clip_prob else None
    if clip:
        info["clip"] = clip
    noisy, target = degrade(clean, rng, noise=noise, rir=rir, snr_db=snr, clip=clip)
    return noisy, target, info


class OnTheFlyDataset(Dataset):
    def __init__(self, speech_csv, noise_csv, rir_csv, split, cfg, length=8000, seed=0):
        sp = pd.read_csv(speech_csv)
        self.speech = sp[sp.split == split].reset_index(drop=True)
        nz = pd.read_csv(noise_csv)
        self.noises = nz[nz.split == split].reset_index(drop=True)
        rr = pd.read_csv(rir_csv)
        self.rirs = rr[rr.split == split].reset_index(drop=True)
        self.n = int(cfg["segment_s"] * 16000)
        conds = cfg["conditions"]  # {name: probability}
        self.cond_names, p = list(conds), np.array(list(conds.values()), dtype=float)
        self.cond_p = p / p.sum()
        self.snr_range = tuple(cfg["snr_range"])
        self.clip_prob = cfg.get("clip_prob", 0.0)
        self.length, self.seed, self.epoch = length, seed, 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        row = self.speech.iloc[int(rng.integers(len(self.speech)))]
        clean = read_segment(row.path, self.n, rng, tile=False)
        cond = self.cond_names[int(rng.choice(len(self.cond_names), p=self.cond_p))]
        noisy, target, info = simulate(clean, rng, cond, self.noises, self.rirs, self.snr_range, self.clip_prob)
        lab = torch.tensor(normalise_labels(info), dtype=torch.float32)
        return torch.from_numpy(noisy), torch.from_numpy(target), lab


class FixedSet(Dataset):
    """Pre-generated mixtures listed in a CSV (columns: noisy, target, snr_db, t60_s, c50_db, ...)."""

    def __init__(self, csv, max_items=None):
        self.df = pd.read_csv(csv)
        if max_items:
            self.df = self.df.iloc[:max_items]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        noisy, _ = sf.read(r.noisy, dtype="float32")
        target, _ = sf.read(r.target, dtype="float32")
        lab = torch.tensor(normalise_labels(r), dtype=torch.float32)
        return torch.from_numpy(noisy), torch.from_numpy(target), lab
