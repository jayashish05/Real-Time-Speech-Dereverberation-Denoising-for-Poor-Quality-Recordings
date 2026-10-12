"""Objective metrics.

Intrusive (need the reference): PESQ (wide-band, ITU-T P.862.2), STOI, SI-SDR, SNR.
Non-intrusive: DNSMOS P.835 (SIG/BAK/OVRL; official ONNX model, CC BY 4.0).
Task metric: WER of wav2vec2-base-960h (greedy CTC) against the LibriSpeech transcript.
"""
import os
import re
from functools import lru_cache

import numpy as np

SR = 16000
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def si_sdr(ref, est, eps=1e-10):
    ref = ref - ref.mean()
    est = est - est.mean()
    t = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    return float(10 * np.log10(np.sum(t ** 2) / (np.sum((est - t) ** 2) + eps) + eps))


def snr(ref, est, eps=1e-10):
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + eps) + eps))


def pesq_wb(ref, est):
    try:
        from pesq import pesq
        return float(pesq(SR, ref, est, "wb"))
    except Exception:  # pesq not installed (hosted build), or no utterance detected
        return float("nan")


def stoi(ref, est):
    from pystoi import stoi as _stoi
    return float(_stoi(ref, est, SR, extended=False))


def intrusive(ref, est, noisy=None):
    n = min(len(ref), len(est))
    ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    out = {"pesq": pesq_wb(ref, est), "stoi": stoi(ref, est), "si_sdr": si_sdr(ref, est), "snr": snr(ref, est)}
    if noisy is not None:
        out["snr_improvement"] = out["snr"] - snr(ref, noisy[:n].astype(np.float64))
    return out


# ------------------------------------------------------------------ DNSMOS P.835

@lru_cache(maxsize=1)
def _dnsmos_session():
    import onnxruntime as ort
    so = ort.SessionOptions()
    # 1 thread by default (evaluation runs a worker pool); the app raises it via DNSMOS_THREADS
    so.intra_op_num_threads = so.inter_op_num_threads = int(os.environ.get("DNSMOS_THREADS", "1"))
    so.log_severity_level = 3
    # CoreML (Apple GPU/ANE) runs the model in ~10 ms vs ~0.4-6 s on CPU; outputs agree to 1e-4
    prov = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(os.path.join(ROOT, "pretrained/dnsmos/sig_bak_ovr.onnx"), so, providers=prov)


def dnsmos(x):
    """Port of microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (non-personalised P.835)."""
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    seg_len = int(9.01 * SR)
    x = np.asarray(x, np.float32)
    while len(x) < seg_len:
        x = np.append(x, x)
    n_hops = int(np.floor(len(x) / SR) - 9.01) + 1
    sess = _dnsmos_session()
    s = []
    for i in range(n_hops):
        seg = x[i * SR: i * SR + seg_len]
        if len(seg) < seg_len:
            continue
        sig, bak, ovr = sess.run(None, {"input_1": seg[None]})[0][0]
        s.append((p_sig(sig), p_bak(bak), p_ovr(ovr)))
    s = np.mean(s, axis=0)
    return {"dnsmos_sig": float(s[0]), "dnsmos_bak": float(s[1]), "dnsmos_ovrl": float(s[2])}


# ------------------------------------------------------------------ ASR / WER

ASR_MODEL = "facebook/wav2vec2-base-960h"


@lru_cache(maxsize=1)
def _asr():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    return Wav2Vec2Processor.from_pretrained(ASR_MODEL), Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).eval()


def transcribe(x):
    import torch
    proc, model = _asr()
    x = np.asarray(x, np.float32)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x)[None]).logits
    return proc.batch_decode(logits.argmax(-1))[0]


def normalise_text(t):
    return re.sub(r"[^A-Z' ]", " ", t.upper()).split()


def wer_counts(ref_text, hyp_text):
    """(edit errors, reference words) so WER can be pooled over a corpus."""
    import jiwer
    r, h = " ".join(normalise_text(ref_text)), " ".join(normalise_text(hyp_text))
    o = jiwer.process_words(r, h or "<empty>")
    return o.substitutions + o.deletions + o.insertions, len(r.split())
