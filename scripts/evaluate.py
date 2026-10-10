"""Evaluate every system on every fixed test set.

Per utterance: PESQ-WB, STOI, SI-SDR, SNR, SNR improvement (vs. the target), DNSMOS P.835, WER
(wav2vec2-base-960h vs. LibriSpeech transcript), processing time. Results are cached in
results/per_utterance.csv, so re-running only computes what is missing.

    python scripts/evaluate.py                       # everything
    python scripts/evaluate.py --systems unprocessed E_dacrn_film --sets noise reverb
    python scripts/evaluate.py --defer-asr      # signal metrics on CPU now, enhanced audio saved ...
    python scripts/evaluate.py --asr-only       # ... then batched WER on the GPU (MPS/CUDA) when it is free
"""
import argparse, os, sys, time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker: parallelism comes from the Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from src.evaluation import metrics as M  # noqa: E402
from src.inference.enhancer import available_models, enhance, load_model, run_system  # noqa: E402
from src.models.classical import CLASSICAL  # noqa: E402
from src.models.dacrn import normalise_labels  # noqa: E402

OUT = "results/per_utterance.csv"
ENH = "results/enhanced"  # enhanced test outputs (used by --asr-only)
ASR_OUT = "results/asr.csv"  # WER rows from --asr-only, merged with per_utterance.csv by load_results()
_models = {}
DEFER_ASR = False


def evaluate_one(job):
    import torch
    torch.set_num_threads(1)
    system, row = job
    noisy, _ = sf.read(row["noisy"], dtype="float32")
    target, _ = sf.read(row["target"], dtype="float32")
    t0 = time.perf_counter()
    if system == "target_reference":  # upper bound: the dereverberated, noise-free target itself
        y = target
    elif system.endswith("oracle"):
        if system not in _models:
            _models[system] = load_model(system)
        y = enhance(_models[system], noisy, cond=normalise_labels(row))[0]
    else:
        y = run_system(system, noisy, _models)
    dt = time.perf_counter() - t0
    r = {"system": system, "id": row["id"], "set": row["set"], "snr_db": row["snr_db"], "t60_s": row["t60_s"],
         "c50_db": row["c50_db"], "proc_s": dt, "dur_s": len(noisy) / 16000}
    r.update(M.intrusive(target, y, noisy))
    r.update(M.dnsmos(y))
    if DEFER_ASR:
        path = f"{ENH}/{system}/{row['id']}.wav"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, np.clip(y, -1, 1), 16000, subtype="PCM_16")
        r.update(wer_err=np.nan, wer_words=np.nan)
    else:
        err, n = M.wer_counts(row["text"], M.transcribe(y))
        r.update(wer_err=err, wer_words=n)
    return r


def _init(defer):
    global DEFER_ASR
    DEFER_ASR = defer


def asr_pass():
    """Fill missing WER from the saved enhanced audio on the GPU if available. Utterances are decoded one at
    a time (no padding): wav2vec2-base normalises over the whole input, so batching with zero padding would
    change its transcripts relative to the per-utterance CPU path used by metrics.transcribe."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    proc = Wav2Vec2Processor.from_pretrained(M.ASR_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(M.ASR_MODEL).eval().to(dev)
    df = pd.read_csv(OUT).drop_duplicates(["system", "id"], keep="last")
    done = set(zip(*pd.read_csv(ASR_OUT)[["system", "id"]].values.T)) if os.path.exists(ASR_OUT) else set()
    todo = [(sy, i) for sy, i in df.loc[df.wer_words.isna(), ["system", "id"]].values if (sy, i) not in done]
    text = pd.read_csv("data/metadata/test.csv").set_index("id").text
    print(f"ASR on {dev}: {len(todo)} utterances")
    rows = []
    for k, (sy, i) in enumerate(todo):
        x = sf.read(f"{ENH}/{sy}/{i}.wav", dtype="float32")[0]
        x = (x - x.mean()) / (x.std() + 1e-7)  # same normalisation as metrics.transcribe
        with torch.inference_mode():
            logits = model(torch.from_numpy(x)[None].to(dev)).logits.cpu()
        e, nw = M.wer_counts(text[i], proc.batch_decode(logits.argmax(-1))[0])
        rows.append({"system": sy, "id": i, "wer_err": e, "wer_words": nw})
        if len(rows) == 200 or k == len(todo) - 1:
            pd.DataFrame(rows).to_csv(ASR_OUT, mode="a", header=not os.path.exists(ASR_OUT), index=False)
            rows = []
            print(f"{k + 1}/{len(todo)}", flush=True)
    print("ASR pass done")


def load_results():
    """per_utterance.csv with WER filled in from the deferred GPU ASR pass."""
    df = pd.read_csv(OUT).drop_duplicates(["system", "id"], keep="last")
    if os.path.exists(ASR_OUT):
        asr = pd.read_csv(ASR_OUT).drop_duplicates(["system", "id"], keep="last").set_index(["system", "id"])
        key = pd.MultiIndex.from_frame(df[["system", "id"]])
        for c in ("wer_err", "wer_words"):
            df[c] = df[c].fillna(pd.Series(asr[c].reindex(key).values, index=df.index))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-csv", default="data/metadata/test.csv")
    ap.add_argument("--defer-asr", action="store_true")
    ap.add_argument("--asr-only", action="store_true")
    a = ap.parse_args()
    if a.asr_only:
        return asr_pass()
    test = pd.read_csv(a.test_csv)
    if a.sets:
        test = test[test.set.isin(a.sets)]
    systems = a.systems or ["unprocessed", "target_reference", *CLASSICAL, *available_models()]
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["system", "id"])
    have = set(zip(done.system, done.id))
    jobs = [(s, r) for s in systems for r in test.to_dict("records") if (s, r["id"]) not in have]
    print(f"{len(jobs)} evaluations to run ({len(have)} cached)")
    os.makedirs("results", exist_ok=True)
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.defer_asr,)) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate_one, jobs, chunksize=4)):
            rows.append(r)
            if len(rows) >= 100 or i == len(jobs) - 1:  # checkpoint often
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                rows = []
                print(f"{i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
