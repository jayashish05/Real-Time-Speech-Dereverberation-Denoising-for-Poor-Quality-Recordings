"""Inference API used by evaluation, the realtime script and the app."""
import glob
import os

import numpy as np
import soundfile as sf
import torch

from src.models.classical import CLASSICAL
from src.models.dacrn import denormalise
from src.preprocessing.dsp import rms_db, to_mono_16k
from src.training.trainer import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs")
DEFAULT_MODEL = "E_dacrn_film"
INPUT_LEVEL_DB = -25.0  # inputs are normalised to the middle of the training level range


def load_audio(path_or_file):
    x, sr = sf.read(path_or_file, dtype="float32", always_2d=True)
    return to_mono_16k(x, sr)


def load_model(name=DEFAULT_MODEL, device="cpu"):
    ck = torch.load(os.path.join(RUNS, name, "best.pt"), map_location=device, weights_only=False)
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    model.info = {"name": name, "params": ck["n_params"], "epoch": ck["epoch"], "val": ck["val"], "cfg": ck["cfg"]}
    return model


def available_models():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RUNS, "*", "best.pt")))


@torch.no_grad()
def enhance(model, x, cond=None):
    """Offline enhancement of a 16 kHz mono signal. Returns (enhanced, per-frame estimates or None)."""
    g = 10 ** ((INPUT_LEVEL_DB - rms_db(x)) / 20)
    w = torch.from_numpy(np.asarray(x, np.float32) * g)[None]
    c = None if cond is None else torch.tensor(cond, dtype=torch.float32)[None]
    y, est = model(w, c)
    return (y[0].numpy() / g).astype(np.float32), None if est is None else est[0].numpy()


def run_system(name, x, model_cache={}):
    """Any system by name: 'unprocessed', classical baselines, or a trained run in runs/."""
    if name == "unprocessed":
        return x
    if name in CLASSICAL:
        return CLASSICAL[name](x)
    if name not in model_cache:
        model_cache[name] = load_model(name)
    m = model_cache[name]
    if m.cond_mode == "oracle":
        raise ValueError("oracle model needs ground-truth labels; use enhance(model, x, cond)")
    return enhance(m, x)[0]


def analyse(est):
    """Summarise per-frame normalised estimates [T, 3] into an interpretable degradation report."""
    if est is None or len(est) == 0:
        return None
    tail = est[len(est) // 3:] if len(est) > 30 else est  # causal estimate settles after ~1 s
    v = {k: float(np.median(a)) for k, a in denormalise(tail).items()}
    v["t60_s"] = max(v["t60_s"], 0.0)
    # thresholds chosen on the validation set (balanced accuracy 0.95 noise / 0.86 reverberation)
    noisy = v["snr_db"] < 24
    reverberant = v["t60_s"] > 0.1 and v["c50_db"] < 22.5
    kind = {(False, False): "clean", (True, False): "noise", (False, True): "reverberation",
            (True, True): "noise + reverberation"}[(noisy, reverberant)]
    severe = v["snr_db"] < 5 or v["t60_s"] > 0.8
    v.update(degradation=kind, severity="severe" if severe else "moderate" if kind != "clean" else "none")
    trace = denormalise(est)
    v["trace"] = {k: np.asarray(a, dtype=float) for k, a in trace.items()}
    return v
