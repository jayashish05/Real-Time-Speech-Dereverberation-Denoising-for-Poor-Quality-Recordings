"""Build metadata (speech / noise / RIR tables with splits and acoustic parameters) and generate the
fixed validation and test sets. Deterministic given --seed. Run after scripts/download_data.py.

Splits (no leakage):
  speech  train = LibriSpeech train-clean-100 subset, val = dev-clean, test = test-clean (disjoint speakers)
  RIRs    SLR28 simulated rooms 1-160 train, 161-180 val, 181-200 test (disjoint rooms);
          SLR28 real RIRs only in the unseen-room test
  noise   SLR28 point-source noises split 80/10/10 by file; DEMAND only in the unseen-noise test
"""
import argparse, glob, os, re, sys, zlib

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import read_segment, simulate  # noqa: E402
from src.preprocessing.dsp import phone_quality, rir_params  # noqa: E402

RIRS_PER_ROOM = 20
MAX_T60 = 1.5  # s; realistic indoor speech range (SLR28 large rooms reach ~7 s)
TEST_SETS = {  # name: (condition, noise split, rir split, snr range, post-processing)
    "clean": ("clean", "test", "test", None, None),
    "noise": ("noise", "test", "test", (-5, 20), None),
    "reverb": ("reverb", "test", "test", None, None),
    "noise_reverb": ("noise_reverb", "test", "test", (-5, 20), None),
    "severe": ("noise_reverb", "test", "test_severe", (-5, 5), None),
    "unseen_noise": ("noise_reverb", "unseen", "test", (-5, 20), None),
    "unseen_room": ("noise_reverb", "test", "unseen", (-5, 20), None),
    "unseen_low_snr": ("noise_reverb", "test", "test", (-10, -5), None),
    "phone": ("noise_reverb", "test", "test", (-5, 20), "phone"),
}


def split_by_hash(name, val=0.1, test=0.1):
    u = (zlib.crc32(name.encode()) % 1000) / 1000
    return "val" if u < val else "test" if u < val + test else "train"


def speech_table(raw):
    rows = []
    for split, folder in [("train", "train-clean-100"), ("val", "dev-clean"), ("test", "test-clean")]:
        for tr in glob.glob(f"{raw}/LibriSpeech/{folder}/*/*/*.trans.txt"):
            d = os.path.dirname(tr)
            for line in open(tr):
                uid, text = line.strip().split(" ", 1)
                p = f"{d}/{uid}.flac"
                if os.path.exists(p):
                    rows.append(dict(path=p, split=split, speaker=uid.split("-")[0], text=text,
                                     duration=sf.info(p).duration))
    df = pd.DataFrame(rows)
    assert not set(df[df.split == "train"].speaker) & set(df[df.split == "test"].speaker)
    return df


def noise_table(raw):
    rows = [dict(path=p, corpus="slr28_pointsource", split=split_by_hash(os.path.basename(p)))
            for p in sorted(glob.glob(f"{raw}/RIRS_NOISES/pointsource_noises/*.wav"))]
    rows += [dict(path=p, corpus="demand", split="unseen")
             for p in sorted(glob.glob(f"{raw}/DEMAND/*/ch01.wav"))]
    df = pd.DataFrame(rows)
    df["duration"] = [sf.info(p).duration for p in df.path]
    return df[df.duration > 0.5].reset_index(drop=True)


def rir_table(raw, rng):
    rows = []
    for size in ("smallroom", "mediumroom", "largeroom"):
        for room_dir in sorted(glob.glob(f"{raw}/RIRS_NOISES/simulated_rirs/{size}/Room*")):
            k = int(re.findall(r"\d+", os.path.basename(room_dir))[0])
            split = "train" if k <= 160 else "val" if k <= 180 else "test"
            files = sorted(glob.glob(f"{room_dir}/*.wav"))
            for p in rng.choice(files, min(RIRS_PER_ROOM, len(files)), replace=False):
                rows.append(dict(path=p, kind=f"sim_{size}", split=split))
    real = sorted(p for p in glob.glob(f"{raw}/RIRS_NOISES/real_rirs_isotropic_noises/*.wav") if "noise" not in os.path.basename(p))
    rows += [dict(path=p, kind="real_" + os.path.basename(p).split("_")[0], split="unseen") for p in real]
    df = pd.DataFrame(rows)
    params = []
    for p in tqdm(df.path, desc="RIR parameters"):
        h, _ = sf.read(p, dtype="float32", always_2d=True)
        params.append(rir_params(h[:, 0]))
    df[["t60_s", "c50_db", "drr_db"]] = np.array(params)
    return df[(df.t60_s > 0.05) & (df.t60_s <= MAX_T60)].reset_index(drop=True)


def make_set(name, spec, utts, noises, rirs, out_dir, rng):
    cond, nsplit, rsplit, snr, post = spec
    nz = noises[noises.split == nsplit]
    if rsplit == "test_severe":  # strongest reverberation among the test rooms
        r = rirs[rirs.split == "test"]
        rr = r[r.t60_s >= r.t60_s.quantile(0.75)]
    else:
        rr = rirs[rirs.split == rsplit]
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)
    rows = []
    for i, u in enumerate(utts.itertuples()):
        clean, _ = sf.read(u.path, dtype="float32")
        noisy, target, info = simulate(clean, rng, cond, nz, rr, snr or (0, 0))
        if post == "phone":
            noisy = phone_quality(noisy)
        base = f"{out_dir}/{name}/{i:04d}"
        sf.write(base + "_noisy.wav", noisy, 16000, subtype="FLOAT")
        sf.write(base + "_target.wav", target, 16000, subtype="FLOAT")
        drr = float(rirs.set_index("path").drr_db.get(info["rir"], np.nan)) if info["rir"] else np.nan
        rows.append(dict(id=f"{name}/{i:04d}", set=name, noisy=base + "_noisy.wav", target=base + "_target.wav",
                         clean=u.path, speaker=u.speaker, text=u.text, drr_db=drr, **info))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--meta", default="data/metadata")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=256)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.meta, exist_ok=True)

    speech = speech_table(a.raw)
    noises = noise_table(a.raw)
    rirs = rir_table(a.raw, rng)
    speech.to_csv(f"{a.meta}/speech.csv", index=False)
    noises.to_csv(f"{a.meta}/noises.csv", index=False)
    rirs.to_csv(f"{a.meta}/rirs.csv", index=False)
    for n, d in [("speech", speech), ("noises", noises), ("rirs", rirs)]:
        print(n, d.groupby("split").size().to_dict())

    # fixed validation set: 6 s crops from dev-clean, training condition mix
    val_sp = speech[speech.split == "val"]
    os.makedirs(f"{a.out}/val", exist_ok=True)
    rows = []
    conds = ["clean", "noise", "reverb", "noise_reverb"]
    for i in range(a.n_val):
        u = val_sp.iloc[int(rng.integers(len(val_sp)))]
        clean = read_segment(u.path, 6 * 16000, rng, tile=False)
        cond = conds[int(rng.choice(4, p=[0.1, 0.3, 0.2, 0.4]))]
        noisy, target, info = simulate(clean, rng, cond, noises[noises.split == "val"], rirs[rirs.split == "val"])
        base = f"{a.out}/val/{i:04d}"
        sf.write(base + "_noisy.wav", noisy, 16000, subtype="FLOAT")
        sf.write(base + "_target.wav", target, 16000, subtype="FLOAT")
        rows.append(dict(noisy=base + "_noisy.wav", target=base + "_target.wav", **info))
    pd.DataFrame(rows).to_csv(f"{a.meta}/val.csv", index=False)

    # fixed test sets: the same utterances (3-10 s, test-clean) in every condition -> paired comparison
    te = speech[(speech.split == "test") & speech.duration.between(3, 10)]
    utts = te.sample(a.n_test, random_state=a.seed).reset_index(drop=True)
    tables = [make_set(name, spec, utts, noises, rirs, f"{a.out}/test", rng) for name, spec in TEST_SETS.items()]
    test = pd.concat(tables, ignore_index=True)
    test.to_csv(f"{a.meta}/test.csv", index=False)
    print(test.groupby("set")[["snr_db", "t60_s", "c50_db"]].mean().round(2))


if __name__ == "__main__":
    main()
