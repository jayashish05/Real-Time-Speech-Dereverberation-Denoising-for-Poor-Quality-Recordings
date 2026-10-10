"""Measure inference cost of every system and the degradation-estimator accuracy.

results/latency.csv   params, model size, GMAC/s, offline RTF (1 CPU thread), streaming ms/frame and RTF,
                      peak process memory, MPS throughput where available
results/estimator.csv DA-CRN utterance-level SNR / T60 / C50 estimates vs. ground truth on the test sets

    python scripts/benchmark.py
"""
import os, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from src.inference.enhancer import analyse, available_models, enhance, load_model, run_system  # noqa: E402
from src.models.classical import CLASSICAL  # noqa: E402
from src.streaming.engine import enhance_streaming  # noqa: E402


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024


def main():
    torch.set_num_threads(1)
    x = sf.read(pd.read_csv("data/metadata/test.csv").query("set == 'noise_reverb'").iloc[0].noisy,
                dtype="float32")[0]
    dur = len(x) / 16000
    rows = []
    for name in CLASSICAL:
        t0 = time.perf_counter()
        for _ in range(3):
            run_system(name, x)
        rows.append({"system": name, "params": 0, "rtf_offline_cpu": (time.perf_counter() - t0) / 3 / dur})
    for name in available_models():
        m = load_model(name)
        with FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 16000), torch.zeros(1, 3) if m.cond_mode == "oracle" else None)
        cond = [0.0, 0.0, 0.0] if m.cond_mode == "oracle" else None
        enhance(m, x, cond)  # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            enhance(m, x, cond)
        r = {"system": name, "params": m.info["params"], "size_mb": m.info["params"] * 4 / 2 ** 20,
             "gmac_per_s": fc.get_total_flops() / 2 / 1e9, "rtf_offline_cpu": (time.perf_counter() - t0) / 3 / dur}
        if m.cond_mode != "oracle":
            _, s = enhance_streaming(m, x)
            r.update(stream_ms_per_frame=s["mean_ms"], stream_p95_ms=s["p95_ms"], stream_rtf=s["rtf"],
                     algorithmic_latency_ms=s["algorithmic_latency_ms"])
        if torch.backends.mps.is_available():
            mm = load_model(name, "mps")
            w = torch.from_numpy(x)[None].to("mps")
            c = torch.zeros(1, 3, device="mps") if m.cond_mode == "oracle" else None
            with torch.no_grad():
                mm(w, c)
                torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    mm(w, c)
                torch.mps.synchronize()
            r["rtf_offline_mps"] = (time.perf_counter() - t0) / 5 / dur
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df["peak_process_mb"] = peak_mb()
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/latency.csv", index=False)
    print(df.round(4).to_string())

    # degradation estimator accuracy (proposed model) on the matched degraded test sets
    m = load_model("E_dacrn_film")
    test = pd.read_csv("data/metadata/test.csv")
    out = []
    for r in test[test.set.isin(["clean", "noise", "reverb", "noise_reverb", "severe", "unseen_room"])].itertuples():
        noisy = sf.read(r.noisy, dtype="float32")[0]
        a = analyse(enhance(m, noisy)[1])
        out.append({"id": r.id, "set": r.set, "snr_db": min(r.snr_db, 40), "t60_s": r.t60_s, "c50_db": min(r.c50_db, 40),
                    "est_snr_db": a["snr_db"], "est_t60_s": a["t60_s"], "est_c50_db": a["c50_db"],
                    "est_degradation": a["degradation"], "condition": r.condition})
    e = pd.DataFrame(out)
    e.to_csv("results/estimator.csv", index=False)
    truth = e.condition.map({"clean": "clean", "noise": "noise", "reverb": "reverberation",
                             "noise_reverb": "noise + reverberation"})
    print("degradation-type accuracy:", (truth == e.est_degradation).mean().round(3))
    print(pd.crosstab(truth, e.est_degradation))


if __name__ == "__main__":
    main()
