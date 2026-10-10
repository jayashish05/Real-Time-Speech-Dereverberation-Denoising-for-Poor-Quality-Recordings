"""Tables (Markdown + CSV) and report figures from results/per_utterance.csv, runs/*/history.csv and
results/latency.csv. Everything plotted is read from measured result files; nothing is hard-coded.

    python scripts/make_figures.py
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

FIG, TAB = "results/figures", "results/tables"
# validated categorical palette (dataviz reference instance, light mode), fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {
    "unprocessed": "Unprocessed", "wiener": "Wiener", "wpe": "WPE", "wpe_wiener": "WPE→Wiener",
    "A_nsnet2": "NSNet2-style", "B_crn_noise_only": "CRN (noise data)", "C_crn_reverb_only": "CRN (reverb data)",
    "D_crn_joint": "CRN (joint data)", "D2_crn_joint_sisdr_only": "CRN joint, SI-SDR only",
    "E_dacrn_film": "DA-CRN (proposed)", "E2_dacrn_multitask": "DA-CRN multitask, no FiLM",
    "E3_dacrn_oracle": "DA-CRN oracle labels", "E4_dacrn_film_clean_rerun": "DA-CRN (clean re-run)",
    "target_reference": "Target (upper bound)",
}
MAIN = ["unprocessed", "wiener", "wpe", "wpe_wiener", "A_nsnet2", "D_crn_joint", "E_dacrn_film"]
ABLATION = ["B_crn_noise_only", "C_crn_reverb_only", "D_crn_joint", "D2_crn_joint_sisdr_only",
            "E2_dacrn_multitask", "E_dacrn_film", "E4_dacrn_film_clean_rerun", "E3_dacrn_oracle"]
MATCHED = ["clean", "noise", "reverb", "noise_reverb", "severe"]
UNSEEN = ["noise_reverb", "unseen_noise", "unseen_room", "unseen_low_snr", "phone"]
METRICS = {"pesq": "PESQ-WB", "stoi": "STOI", "si_sdr": "SI-SDR (dB)", "snr_improvement": "SNR improvement (dB)",
           "wer": "WER (%)", "dnsmos_ovrl": "DNSMOS OVRL"}

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def agg(df, by):
    g = df.groupby(by)
    out = g[["pesq", "stoi", "si_sdr", "snr_improvement", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]].mean()
    out["wer"] = 100 * g.wer_err.sum() / g.wer_words.sum()  # corpus-level WER
    out["n"] = g.size()
    return out


def present(systems, df):
    have = set(df.system)
    return [s for s in systems if s in have]


def grouped_bars(tab, systems, groups, metric, path, title):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(groups)
    for j, g in enumerate(groups):
        vals = [tab.loc[(s, g), metric] if (s, g) in tab.index else np.nan for s in systems]
        ax.bar(np.arange(len(systems)) + (j - (len(groups) - 1) / 2) * w, vals, w * 0.9, color=PAL[j],
               label=g.replace("_", " + " if g == "noise_reverb" else " "), edgecolor=SURF, linewidth=1)
    ax.set_xticks(np.arange(len(systems)), [NAMES.get(s, s) for s in systems], rotation=20, ha="right")
    ax.set_ylabel(METRICS[metric])
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(groups), loc="upper left", bbox_to_anchor=(0, -0.32), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(df, floatfmt=".2f"):
    return df.to_markdown(floatfmt=floatfmt)


def binned(df, systems, col, bins, metric, path, xlabel, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = df.copy()
    d["bin"] = pd.cut(d[col], bins)
    for j, s in enumerate(systems):
        g = d[d.system == s].groupby("bin", observed=True)
        y = g[metric].mean() if metric != "wer" else 100 * g.wer_err.sum() / g.wer_words.sum()
        x = [b.mid for b in y.index]
        ax.plot(x, y.values, "-o", color=PAL[j], lw=2, ms=5, label=NAMES.get(s, s))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRICS[metric])
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def curves():
    hs = {os.path.basename(os.path.dirname(p)): pd.read_csv(p) for p in sorted(glob.glob("runs/*/history.csv"))}
    for key, label, fname in [("train_loss", "Training loss", "fig05_training_loss.png"),
                              ("val_loss", "Validation loss", "fig06_validation_loss.png")]:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for j, (name, h) in enumerate(hs.items()):
            ax.plot(h.epoch + 1, h[key], "-o", ms=4, lw=2, color=PAL[j % len(PAL)], label=NAMES.get(name, name))
        ax.set_xlabel("epoch (500 steps × 16 × 4 s)")
        ax.set_ylabel(label + " (lower is better)")
        ax.set_title(label + " per experiment", loc="left")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{FIG}/{fname}", dpi=160)
        plt.close(fig)
    rows = []
    for name, h in hs.items():
        b = h.loc[h.val_loss.idxmin()]
        rows.append({"experiment": name, "epochs": len(h), "best_epoch": int(b.epoch) + 1,
                     "best_val_loss": b.val_loss, "val_si_sdr_dB": b.val_si_sdr,
                     "train_time_min": h.time_s.sum() / 60})
    if rows:
        t = pd.DataFrame(rows).set_index("experiment")
        t.to_csv(f"{TAB}/training_summary.csv")
        open(f"{TAB}/training_summary.md", "w").write(md_table(t))


def examples():
    """Waveform + spectrogram of one noise+reverb test utterance: noisy / enhanced / target."""
    test = pd.read_csv("data/metadata/test.csv")
    row = test[test.set == "noise_reverb"].iloc[3]
    from src.inference.enhancer import enhance, load_model
    noisy, _ = sf.read(row.noisy, dtype="float32")
    target, _ = sf.read(row.target, dtype="float32")
    enh = enhance(load_model("E_dacrn_film"), noisy)[0]
    sigs = [("Original (noisy + reverberant)", noisy), ("Enhanced (DA-CRN)", enh), ("Target (clean, direct path)", target)]
    t = np.arange(len(noisy)) / 16000
    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)
    for ax, (name, x) in zip(axs, sigs):
        ax.plot(t, x, color=PAL[0], lw=0.5)
        ax.set_title(name, loc="left")
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(f"Waveforms — SNR {row.snr_db:.1f} dB, T60 {row.t60_s:.2f} s", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig01_waveforms.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    for ax, (name, x) in zip(axs, sigs):
        S = 20 * np.log10(np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, 512)[::128] * np.hanning(512), axis=1)).T + 1e-6)
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=S.max() - 80, vmax=S.max(),
                       extent=[0, len(x) / 16000, 0, 8])
        ax.set_title(name, loc="left")
        ax.set_ylabel("kHz")
        ax.grid(False)
    axs[-1].set_xlabel("time (s)")
    fig.colorbar(im, ax=axs, label="dB", shrink=0.6)
    fig.savefig(f"{FIG}/fig02_spectrograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def estimator_scatter():
    """Degradation estimator accuracy on the matched test sets (DA-CRN, utterance-level median)."""
    path = "results/estimator.csv"
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (k, lab) in zip(axs, [("snr_db", "SNR (dB)"), ("t60_s", "T60 (s)"), ("c50_db", "C50 (dB)")]):
        ax.scatter(d[k], d["est_" + k], s=10, color=PAL[0], alpha=0.6, edgecolor="none")
        lo, hi = d[k].min(), d[k].max()
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
        mae = (d[k] - d["est_" + k]).abs().mean()
        ax.set_title(f"{lab}: MAE {mae:.2f}", loc="left")
        ax.set_xlabel("true")
        ax.set_ylabel("estimated")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig13_degradation_estimator.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    curves()
    if os.path.exists("runs/E_dacrn_film/best.pt"):
        examples()
    estimator_scatter()
    if not os.path.exists("results/per_utterance.csv"):
        return
    from scripts.evaluate import load_results
    df = load_results()
    missing = df.wer_words.isna().sum()
    if missing:
        print(f"warning: {missing} rows still lack WER (run scripts/evaluate.py --asr-only)")
    tab = agg(df, ["system", "set"])
    tab.to_csv(f"{TAB}/all_systems_by_set.csv")

    main_sys, abl_sys = present(MAIN, df), present(ABLATION, df)
    # main results table: matched conditions
    for metric in ["pesq", "stoi", "si_sdr", "wer", "snr_improvement", "dnsmos_ovrl"]:
        t = tab[metric].unstack("set").reindex(index=present(MAIN + ["target_reference"], df))
        t = t[[c for c in MATCHED + UNSEEN[1:] if c in t.columns]]
        t.index = [NAMES.get(s, s) for s in t.index]
        open(f"{TAB}/{metric}_by_condition.md", "w").write(md_table(t, ".1f" if metric in ("wer", "si_sdr", "snr_improvement") else ".3f"))
    # compact headline table on the joint condition
    allsys = present(list(dict.fromkeys(MAIN + ABLATION + ["target_reference"])), df)
    hl = tab.xs("noise_reverb", level="set").reindex(allsys)
    lat = pd.read_csv("results/latency.csv").set_index("system") if os.path.exists("results/latency.csv") else None
    cols = ["pesq", "stoi", "si_sdr", "snr_improvement", "wer", "dnsmos_ovrl"]
    hl = hl[cols]
    if lat is not None:
        hl = hl.join(lat[["params", "rtf_offline_cpu", "stream_ms_per_frame"]], how="left")
    hl.index = [NAMES.get(s, s) for s in hl.index]
    hl.to_csv(f"{TAB}/headline_noise_reverb.csv")
    open(f"{TAB}/headline_noise_reverb.md", "w").write(md_table(hl, ".3f"))
    # ablation table (mean over the four degraded matched conditions)
    abl = agg(df[df.set.isin(MATCHED[1:])], ["system"]).reindex(abl_sys)[cols]
    abl.index = [NAMES.get(s, s) for s in abl.index]
    open(f"{TAB}/ablation.md", "w").write(md_table(abl, ".3f"))
    # generalisation table
    gen = tab["pesq"].unstack("set").reindex(allsys[:-1])[[c for c in UNSEEN if c in tab.index.get_level_values(1)]]
    gen.index = [NAMES.get(s, s) for s in gen.index]
    open(f"{TAB}/generalisation_pesq.md", "w").write(md_table(gen, ".3f"))
    gw = tab["wer"].unstack("set").reindex(allsys[:-1])[[c for c in UNSEEN if c in tab.index.get_level_values(1)]]
    gw.index = [NAMES.get(s, s) for s in gw.index]
    open(f"{TAB}/generalisation_wer.md", "w").write(md_table(gw, ".1f"))

    for metric in ("pesq", "wer", "stoi"):
        t = tab[metric].unstack("set").reindex(["unprocessed"] + abl_sys)[MATCHED[1:]]
        t.index = [NAMES.get(x, x) for x in t.index]
        open(f"{TAB}/ablation_{metric}_by_condition.md", "w").write(md_table(t, ".1f" if metric == "wer" else ".3f"))
    significance(df)

    # figures
    for k, (metric, fname) in enumerate([("pesq", "fig07_pesq.png"), ("stoi", "fig08_stoi.png"),
                                          ("si_sdr", "fig09_si_sdr.png"), ("wer", "fig10_wer.png"),
                                          ("snr_improvement", "fig12_snr_improvement.png")]):
        grouped_bars(tab, main_sys, [c for c in MATCHED if c != "clean" or metric != "snr_improvement"], metric,
                     f"{FIG}/{fname}", f"{METRICS[metric]} by system and test condition (matched)")
    grouped_bars(tab, allsys[:-1], UNSEEN, "pesq", f"{FIG}/fig14_generalisation_pesq.png",
                 "PESQ-WB: matched noise + reverb vs. unseen conditions")
    grouped_bars(tab, abl_sys, ["noise", "reverb", "noise_reverb", "severe"], "pesq", f"{FIG}/fig15_ablation_pesq.png",
                 "Ablation: training data, loss and conditioning (PESQ-WB)")
    grouped_bars(tab, abl_sys, ["noise", "reverb", "noise_reverb", "severe"], "wer", f"{FIG}/fig16_ablation_wer.png",
                 "Ablation: training data, loss and conditioning (WER)")
    key = present(["unprocessed", "wpe_wiener", "D_crn_joint", "E_dacrn_film"], df)
    nr = df[df.set.isin(["noise", "noise_reverb", "unseen_low_snr"])]
    binned(nr, key, "snr_db", [-10, -5, 0, 5, 10, 15, 20], "pesq", f"{FIG}/fig17_pesq_vs_snr.png",
           "input SNR (dB)", "PESQ-WB vs. input SNR")
    binned(nr, key, "snr_db", [-10, -5, 0, 5, 10, 15, 20], "wer", f"{FIG}/fig18_wer_vs_snr.png",
           "input SNR (dB)", "WER vs. input SNR")
    rv = df[df.set.isin(["reverb", "noise_reverb", "severe"])]
    binned(rv, key, "t60_s", [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6], "pesq", f"{FIG}/fig19_pesq_vs_t60.png",
           "T60 (s)", "PESQ-WB vs. reverberation time")
    binned(rv, key, "t60_s", [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6], "stoi", f"{FIG}/fig20_stoi_vs_t60.png",
           "T60 (s)", "STOI vs. reverberation time")
    if lat is not None:
        latency_fig(lat)
    print(open(f"{TAB}/headline_noise_reverb.md").read())


PAIRS = [("E_dacrn_film", "D_crn_joint"), ("E4_dacrn_film_clean_rerun", "D_crn_joint"),
         ("E4_dacrn_film_clean_rerun", "E_dacrn_film"), ("E2_dacrn_multitask", "D_crn_joint"),
         ("E3_dacrn_oracle", "D_crn_joint"), ("E3_dacrn_oracle", "E_dacrn_film"), ("D_crn_joint", "D2_crn_joint_sisdr_only"),
         ("D_crn_joint", "B_crn_noise_only"), ("D_crn_joint", "C_crn_reverb_only"), ("D_crn_joint", "A_nsnet2"),
         ("E_dacrn_film", "wpe_wiener"), ("E_dacrn_film", "unprocessed")]


def significance(df, sets=MATCHED[1:], n_boot=2000, seed=0):
    """Paired bootstrap over utterances (same utterances for every system): mean difference a-b and 95% CI.
    WER differences are corpus-level (errors / words) on each resample."""
    d = df[df.set.isin(sets)].set_index(["system", "id"])
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in PAIRS:
        if a not in d.index.get_level_values(0) or b not in d.index.get_level_values(0):
            continue
        A, B = d.loc[a], d.loc[b].reindex(d.loc[a].index)
        idx = rng.integers(0, len(A), (n_boot, len(A)))
        r = {"comparison": f"{NAMES.get(a, a)} − {NAMES.get(b, b)}"}
        for m in ("pesq", "stoi", "si_sdr"):
            x = (A[m] - B[m]).to_numpy()
            ok = ~np.isnan(x)
            x = x[ok]
            bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r[m] = f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if lo > 0 or hi < 0 else "")
        ea, wa, eb = A.wer_err.to_numpy(), A.wer_words.to_numpy(), B.wer_err.to_numpy()
        bs = 100 * (ea[idx].sum(1) - eb[idx].sum(1)) / wa[idx].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r["wer (pts)"] = f"{100 * (ea.sum() - eb.sum()) / wa.sum():+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else "")
        rows.append(r)
    t = pd.DataFrame(rows).set_index("comparison")
    open(f"{TAB}/significance.md", "w").write(
        f"Paired bootstrap ({n_boot} resamples) over the {len(sets)} degraded matched test sets "
        f"({', '.join(sets)}; 400 utterance pairs). Mean difference a − b [95% CI]; * = CI excludes 0. "
        "Higher is better for PESQ/STOI/SI-SDR, lower for WER.\n\n" + t.to_markdown())


def latency_fig(lat):
    lat = lat.reindex([s for s in dict.fromkeys(MAIN + ABLATION) if s in lat.index and not s.endswith("oracle")])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(lat))
    ax.barh(y, lat.rtf_offline_cpu, 0.6, color=PAL[0], label="offline, whole file")
    ax.barh(y + 0.3, lat.stream_rtf.fillna(0), 0.3, color=PAL[1], label="streaming, frame by frame")
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.text(1.0, len(lat) - 0.4, " real-time limit", color=INK2, fontsize=8)
    ax.set_yticks(y + 0.15, [NAMES.get(s, s) for s in lat.index])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("real-time factor on 1 CPU thread (lower is faster)")
    ax.set_title("Inference cost (Apple M2, single thread)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig11_latency.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
