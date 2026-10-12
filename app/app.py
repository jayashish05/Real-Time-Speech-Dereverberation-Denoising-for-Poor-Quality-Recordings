"""Real-Time Speech Dereverberation and Denoising for Poor-Quality Recordings: demo application.

    streamlit run app/app.py
"""
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("DNSMOS_THREADS", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from src.evaluation import metrics as M  # noqa: E402
from src.inference.enhancer import (DEFAULT_MODEL, analyse, available_models, enhance, load_audio,  # noqa: E402
                                    load_model, run_system)
from src.models.classical import CLASSICAL  # noqa: E402
from src.models.dacrn import denormalise  # noqa: E402
from src.streaming.engine import StreamingEnhancer  # noqa: E402

st.set_page_config(page_title="Speech Enhancement", page_icon="🎙️", layout="wide")
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NAMES = {"wiener": "Wiener filter (classical)", "wpe": "WPE dereverberation (classical)",
         "wpe_wiener": "WPE → Wiener (classical)"}


# ------------------------------------------------------------------ cached resources
@st.cache_resource
def get_model(name):
    torch.set_num_threads(2)
    return load_model(name)


def model_options():
    ms = [m for m in available_models() if not m.endswith("oracle")]
    ms.sort(key=lambda m: m != DEFAULT_MODEL)
    return ms + list(CLASSICAL)


def label(name):
    if name in NAMES:
        return NAMES[name]
    return name + (" (proposed)" if name == DEFAULT_MODEL else "")


def process(system, x):
    """Returns enhanced signal, per-frame estimates (DA-CRN only), seconds of compute."""
    t0 = time.perf_counter()
    if system in CLASSICAL:
        y, est = run_system(system, x), None
    else:
        y, est = enhance(get_model(system), x)
    return y, est, time.perf_counter() - t0


# ------------------------------------------------------------------ plotting
def fig_wave(signals):
    fig, axs = plt.subplots(len(signals), 1, figsize=(10, 1.6 * len(signals)), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (name, x), c in zip(axs, signals, PAL):
        ax.plot(np.arange(len(x)) / 16000, x, lw=0.5, color=c)
        ax.set_title(name, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def fig_spec(signals):
    fig, axs = plt.subplots(len(signals), 1, figsize=(10, 2.2 * len(signals)), sharex=True)
    axs = np.atleast_1d(axs)
    ref = None
    for ax, (name, x) in zip(axs, signals):
        if len(x) < 512:
            x = np.pad(x, (0, 512 - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::128] * np.hanning(512)
        S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-7)
        ref = S.max() if ref is None else ref
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=ref - 80, vmax=ref, extent=[0, len(x) / 16000, 0, 8])
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("kHz")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def fig_trace(a):
    tr = a["trace"]
    t = np.arange(len(tr["snr_db"])) * 256 / 16000
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.6))
    for ax, (k, lab), c in zip(axs, [("snr_db", "estimated SNR (dB)"), ("t60_s", "estimated T60 (s)"),
                                      ("c50_db", "estimated C50 (dB)")], PAL):
        ax.plot(t, tr[k], color=c, lw=1.5)
        ax.axhline(a[k], color="#52514e", ls="--", lw=1)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def wav_bytes(x):
    b = io.BytesIO()
    sf.write(b, np.clip(x, -1, 1), 16000, format="WAV", subtype="PCM_16")
    return b.getvalue()


def degradation_panel(a):
    if a is None:
        st.info("Degradation analysis comes from DA-CRN's estimator head. Choose a DA-CRN model to see it.")
        return
    c = st.columns(5)
    c[0].metric("Detected degradation", a["degradation"])
    c[1].metric("Severity", a["severity"])
    c[2].metric("Estimated SNR", f"{a['snr_db']:.1f} dB")
    c[3].metric("Estimated T60", f"{a['t60_s']:.2f} s")
    c[4].metric("Estimated C50", f"{a['c50_db']:.1f} dB")
    st.pyplot(fig_trace(a))
    st.caption("Estimates are produced causally, frame by frame, by the degradation head and fed to the FiLM "
               "layers that adapt the enhancement. Dashed line: utterance summary (median after the first third). "
               "Labels: noise if SNR < 24 dB; reverberation if T60 > 0.1 s and C50 < 22.5 dB (thresholds calibrated on the validation set).")


# ------------------------------------------------------------------ pages
def page_home():
    st.title("Real-Time Speech Dereverberation and Denoising for Poor-Quality Recordings")
    st.markdown(
        "This system restores speech degraded by **background noise**, **room reverberation** and "
        "**recording-chain clipping**. The proposed **DA-CRN** (degradation-aware convolutional recurrent "
        "network) is causal and runs frame by frame on a CPU. It estimates the acoustic condition (SNR, T60, "
        "C50) online, reports it to the user, and uses it to modulate its own processing (FiLM conditioning). "
        "The measured effect of each component is on the Analysis page and in docs/experiments.md.")
    st.graphviz_chart("""
    digraph G { rankdir=TB; nodesep=0.3; ranksep=0.25; node [shape=box, style="rounded,filled", fillcolor="#f0efec", color="#c3c2b7", fontname="Helvetica", fontsize=11];
      mic [label="Microphone / file\\n16 kHz"]; stft [label="STFT\\n32 ms window, 16 ms hop"];
      enc [label="Causal conv encoder\\n5 layers"]; est [label="Degradation estimator\\nGRU → SNR, T60, C50", fillcolor="#cde2fb"];
      film [label="FiLM\\nγ·h + β", fillcolor="#cde2fb"]; gru [label="2-layer GRU\\n256 units"];
      dec [label="Decoder + skips\\n→ bounded complex mask"]; istft [label="iSTFT\\noverlap-add"]; out [label="Enhanced speech"];
      ana [label="Degradation report\\n(noise / reverb / both, severity)", shape=note];
      mic -> stft -> enc -> film -> gru -> dec -> istft -> out; enc -> est -> film; est -> ana; stft -> dec [style=dashed, label="mask × X"]; }
    """)
    st.markdown("""
| Stage | What happens |
|---|---|
| Degradation simulation (training) | LibriSpeech clean speech ⊛ SLR28 room impulse response + SLR28 noise at -5…20 dB SNR, random level, random clipping |
| Degradation analysis | causal estimator predicts SNR, T60 and C50 every 16 ms |
| Adaptive enhancement | FiLM layers scale and shift bottleneck features according to the estimate |
| Reconstruction | tanh-bounded complex ratio mask, inverse STFT with overlap-add (streamable) |

Use the sidebar to enhance a file, run live microphone enhancement, inspect the analysis, or read the model card.
""")


def page_file():
    st.header("File enhancement")
    src = st.radio("Input", ["Upload a recording", "Example from the test set (has a clean reference)"], horizontal=True)
    x = ref = text = key = None
    if src.startswith("Upload"):
        f = st.file_uploader("WAV / MP3 / FLAC / OGG (any sample rate; converted to 16 kHz mono)",
                             type=["wav", "mp3", "flac", "ogg"])
        r = st.file_uploader("Optional clean reference (enables PESQ / STOI / SI-SDR)", type=["wav", "flac", "mp3"])
        text = st.text_input("Optional transcript (enables WER)")
        if f is not None:
            try:
                x = load_audio(f)
                ref = load_audio(r) if r is not None else None
            except Exception as e:
                st.error(f"Could not decode the file: {e}")
                return
            key = ("upload", f.name, f.size, r.name if r is not None else None)
    else:
        test = pd.read_csv("data/metadata/test.csv")
        test = test[test.noisy.map(os.path.exists)]  # a clone ships 3 examples per condition; the full set needs prepare_data
        if test.empty:
            st.warning("Test-set audio is not on this machine. Create it with "
                       "`python scripts/download_data.py` and `python scripts/prepare_data.py`, or upload a file.")
            return
        c1, c2 = st.columns(2)
        sets = list(test.set.unique())
        s = c1.selectbox("Condition", sets, index=sets.index("noise_reverb") if "noise_reverb" in sets else 0)
        sub = test[test.set == s].reset_index(drop=True)
        i = c2.slider("Utterance", 0, len(sub) - 1, 0) if len(sub) > 1 else 0
        row = sub.iloc[i]
        st.caption(f"Simulated: SNR {row.snr_db:.1f} dB · T60 {row.t60_s:.2f} s · C50 {row.c50_db:.1f} dB · "
                   f"speaker {row.speaker} (test speakers were never seen in training)")
        x = sf.read(row.noisy, dtype="float32")[0]
        ref = sf.read(row.target, dtype="float32")[0]
        text, key = row.text, ("example", row.id)
    if x is None:
        return
    if len(x) > 16000 * 120:
        st.warning("Long file: only the first 120 s are processed.")
        x = x[: 16000 * 120]
    system = st.selectbox("Enhancement system", model_options(), format_func=label)
    key = (*key, system)
    if st.button("Enhance", type="primary"):
        with st.spinner("Enhancing…"):
            y, est, dt = process(system, x)
        st.session_state["last"] = {"key": key, "x": x, "y": y, "ref": ref, "text": text, "system": system,
                                    "analysis": analyse(est), "dt": dt}
    last = st.session_state.get("last")
    if not last or last.get("key") != key:  # nothing enhanced yet for this input + system
        return
    if last["text"] != text:  # transcript typed after enhancing
        last["text"] = text
        last.pop("metrics", None)
    x, y, ref, dt = last["x"], last["y"], last["ref"], last["dt"]

    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.audio(wav_bytes(x), format="audio/wav")
    c2.subheader("Enhanced")
    c2.audio(wav_bytes(y), format="audio/wav")
    c2.download_button("Download enhanced WAV", wav_bytes(y), "enhanced.wav", "audio/wav")
    st.caption(f"Processing time {dt:.2f} s for {len(x) / 16000:.1f} s of audio (real-time factor {dt / (len(x) / 16000):.3f}).")
    sig = [("Original", x), ("Enhanced", y)] + ([("Clean reference", ref)] if ref is not None else [])
    t1, t2 = st.tabs(["Waveforms", "Spectrograms"])
    t1.pyplot(fig_wave(sig))
    t2.pyplot(fig_spec(sig))
    st.subheader("Degradation analysis")
    degradation_panel(last["analysis"])
    st.subheader("Metrics")
    show_metrics(last)


LABELS = {"pesq": "PESQ-WB", "stoi": "STOI", "si_sdr": "SI-SDR (dB)", "snr": "SNR (dB)", "dnsmos_sig": "DNSMOS SIG",
          "dnsmos_bak": "DNSMOS BAK", "dnsmos_ovrl": "DNSMOS OVRL", "wer": "WER (%)"}
ASR_DEFAULT = os.path.isdir("data/raw")  # on where the datasets are (a dev machine); off on a hosted clone


def show_metrics(last, where="file"):
    """Metrics for the last enhanced input; cached in `last` so page reruns do not recompute them."""
    x, y, ref, text = last["x"], last["y"], last["ref"], last["text"]
    asr = st.toggle("Speech recognition (ASR transcript and WER)", value=ASR_DEFAULT, key=f"asr_{where}",
                    help="Loads wav2vec2-base-960h (a 360 MB model). Leave off on free hosting with little memory.")
    m = last.get("metrics")
    if m is None or (asr and not m["hyps"]):
        with st.spinner("Computing metrics…"):
            cols, hyps = {}, {}
            for name, s in [("Original", x), ("Enhanced", y)]:
                r = M.dnsmos(s)
                if ref is not None:
                    r.update(M.intrusive(ref, s))
                if asr:
                    hyps[name] = M.transcribe(s)
                    if text:
                        e, n = M.wer_counts(text, hyps[name])
                        r["wer"] = 100 * e / max(n, 1)
                cols[name] = r
        m = last["metrics"] = {"cols": cols, "hyps": hyps}
    df = pd.DataFrame(m["cols"])
    df["Change"] = df["Enhanced"] - df["Original"]
    no_pesq = ref is not None and df.loc["pesq"].isna().all()
    df = df.dropna(how="all")
    if not asr and "wer" in df.index:
        df = df.drop(index="wer")
    df.index = [LABELS.get(k, k) for k in df.index]
    st.dataframe(df.style.format("{:.2f}"), width="stretch")
    if ref is not None:
        st.metric("SNR improvement", f"{df.loc['SNR (dB)', 'Change']:+.1f} dB")
    if asr and m["hyps"]:
        h = m["hyps"]
        st.markdown(f"**ASR (wav2vec2) on original:** {h['Original'] or '∅'}  \n**ASR on enhanced:** {h['Enhanced'] or '∅'}"
                    + (f"  \n**Reference transcript:** {text}" if text else ""))
    st.caption("PESQ-WB / STOI / SI-SDR / SNR need a clean reference. DNSMOS P.835 (SIG / BAK / OVRL) is non-intrusive. "
               "WER uses wav2vec2-base-960h." + (" PESQ is not installed on this server." if no_pesq else ""))


def page_live():
    st.header("Live enhancement")
    st.markdown("**Mode 1: true real-time loop.** The microphone is processed in 16 ms blocks inside the audio "
                "callback, and the enhanced audio goes straight to the output device. Use headphones to avoid "
                "feedback. This uses the audio devices of the machine running the app (local use).")
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins = {i: d["name"] for i, d in enumerate(devs) if d["max_input_channels"] > 0}
        outs = {i: d["name"] for i, d in enumerate(devs) if d["max_output_channels"] > 0}
    except Exception:  # no PortAudio / no devices
        ins, outs = {}, {}
    if not ins:
        st.info("This machine has no microphone (normal on a hosted server), so the real-time loop is not "
                "available here. Run the app on your own computer for Mode 1, or use Mode 2 below.")
    models = [m for m in model_options() if m not in CLASSICAL]
    if ins and models:
        c = st.columns(4)
        mdl = c[0].selectbox("Model", models, format_func=label)
        di = c[1].selectbox("Input", list(ins), format_func=ins.get, index=list(ins).index(sd.default.device[0]) if sd.default.device[0] in ins else 0)
        do = c[2].selectbox("Output", list(outs), format_func=outs.get, index=list(outs).index(sd.default.device[1]) if sd.default.device[1] in outs else 0)
        gain = c[3].slider("Input gain (dB)", -10, 30, 0)
        b = st.columns(4)
        running = st.session_state.get("live") is not None
        if b[0].button("▶ Start", disabled=running, type="primary"):
            from src.streaming.live import LiveSession
            try:
                st.session_state["live"] = LiveSession(get_model(mdl), di, do, gain).start()
                st.rerun()
            except Exception as e:
                st.error(f"Could not open the audio stream: {e}")
        if b[1].button("■ Stop", disabled=not running):
            sess = st.session_state.pop("live")
            sess.stop()
            st.session_state["live_rec"] = (*sess.recordings(), sess.stats())
            st.rerun()
        if running:
            st.session_state["live"].bypass = b[2].toggle("Bypass (hear the original)", value=False)
            live_stats()
        if "live_rec" in st.session_state and not running:
            xin, yout, s = st.session_state["live_rec"]
            st.success(f"Session: {s['recorded_s']:.1f} s · {s['mean_ms']:.2f} ms compute per 16 ms frame "
                       f"(RTF {s['rtf']:.3f}) · end-to-end latency ≈ {s['end_to_end_latency_ms']:.0f} ms · {s['xruns']} late blocks")
            if len(xin):
                c1, c2 = st.columns(2)
                c1.audio(wav_bytes(xin), format="audio/wav")
                c1.caption("Microphone input")
                c2.audio(wav_bytes(yout), format="audio/wav")
                c2.caption("Enhanced output (as played)")
                st.pyplot(fig_spec([("Microphone input", xin), ("Enhanced (streamed)", yout)]))
    st.divider()
    st.markdown("**Mode 2: browser recording.** Record in the browser, then the clip is pushed through the *same "
                "streaming engine* frame by frame (not the offline path) and per-frame compute time is measured. "
                "Use this when the app runs on a remote server.")
    clip = st.audio_input("Record a clip")
    if clip is not None and models:
        mdl2 = st.selectbox("Model ", models, format_func=label, key="m2")
        x = load_audio(clip)
        se = StreamingEnhancer(get_model(mdl2))
        torch.set_num_threads(1)
        y = np.concatenate([se.process(x[i:i + 256]) for i in range(0, len(x), 256)])
        s = se.stats()
        c = st.columns(4)
        c[0].metric("Compute / 16 ms frame", f"{s['mean_ms']:.2f} ms")
        c[1].metric("p95 compute", f"{s['p95_ms']:.2f} ms")
        c[2].metric("Real-time factor", f"{s['rtf']:.3f}")
        c[3].metric("Algorithmic latency", f"{s['algorithmic_latency_ms']:.0f} ms")
        c1, c2 = st.columns(2)
        c1.audio(wav_bytes(x), format="audio/wav")
        c2.audio(wav_bytes(y), format="audio/wav")
        st.pyplot(fig_spec([("Recorded", x), ("Enhanced (streaming engine)", y)]))
        a = None
        if se.estimate is not None:
            a = analyse(enhance(get_model(mdl2), x)[1])
        st.session_state["last"] = {"x": x, "y": y, "ref": None, "text": None, "system": mdl2, "analysis": a, "dt": None}
        degradation_panel(a)


@st.fragment(run_every=1.0)
def live_stats():
    sess = st.session_state.get("live")
    if sess is None:
        return
    s = sess.stats()
    c = st.columns(6)
    c[0].metric("Compute / frame", f"{s['mean_ms']:.2f} ms", help="frame = 16 ms of audio")
    c[1].metric("p95 compute", f"{s['p95_ms']:.2f} ms")
    c[2].metric("Real-time factor", f"{s['rtf']:.3f}")
    c[3].metric("End-to-end latency", f"{s['end_to_end_latency_ms']:.0f} ms",
                help=f"32 ms algorithmic + compute + device buffers ({s['device_latency_ms']:.0f} ms)")
    c[4].metric("Late blocks", s["xruns"])
    c[5].metric("Input level", f"{s['level_in_db']:.0f} dBFS")
    if s["level_in_db"] < -90 and s["recorded_s"] > 1:
        st.warning("The microphone is delivering digital silence. On macOS, allow microphone access for the app "
                   "running Streamlit (Terminal / IDE) in System Settings → Privacy & Security → Microphone, then restart it.")
    est = sess.enh.estimate
    if est is not None:
        v = {k: float(a) for k, a in denormalise(np.asarray(est)).items()}
        st.caption(f"Live degradation estimate: SNR {v['snr_db']:.1f} dB · T60 {max(v['t60_s'], 0):.2f} s · "
                   f"C50 {v['c50_db']:.1f} dB · recorded {s['recorded_s']:.0f} s")


def page_analysis():
    st.header("Analysis")
    last = st.session_state.get("last")
    if last is None:
        st.info("Enhance a file (or a recording) first. Its analysis will appear here.")
    else:
        st.subheader(f"Last input: {label(last['system'])}")
        degradation_panel(last["analysis"])
        show_metrics(last, where="analysis")
    st.subheader("Test-set results (measured, see docs/experiments.md)")
    tdir = "results/tables"
    if os.path.isdir(tdir):
        for f, title in [("headline_noise_reverb.md", "Noise + reverberation test set"),
                         ("pesq_by_condition.md", "PESQ-WB by condition"), ("wer_by_condition.md", "WER (%) by condition"),
                         ("ablation.md", "Ablation (mean over degraded matched conditions)"),
                         ("significance.md", "Paired significance tests (bootstrap 95 % CI)"),
                         ("generalisation_pesq.md", "Generalisation: PESQ-WB on unseen conditions")]:
            if os.path.exists(f"{tdir}/{f}"):
                with st.expander(title, expanded=f.startswith("headline")):
                    st.markdown(open(f"{tdir}/{f}").read())
    figs = sorted(p for p in os.listdir("results/figures")) if os.path.isdir("results/figures") else []
    if figs:
        sel = st.selectbox("Figure", figs)
        st.image(f"results/figures/{sel}")


def page_model():
    st.header("Model information")
    name = st.selectbox("Model", [m for m in model_options() if m not in CLASSICAL], format_func=label)
    m = get_model(name)
    cfg = m.info["cfg"]
    c = st.columns(4)
    c[0].metric("Parameters", f"{m.info['params']:,}")
    c[1].metric("Size (fp32)", f"{m.info['params'] * 4 / 2 ** 20:.1f} MB")
    c[2].metric("Best epoch", m.info["epoch"] + 1)
    c[3].metric("Validation SI-SDR", f"{m.info['val']['si_sdr']:.2f} dB")
    st.markdown(f"""
| | |
|---|---|
| Architecture | {'NSNet2-style GRU magnitude mask' if cfg['model']['name'] == 'nsnet2' else f"Causal CRN, encoder channels {cfg['model']['channels']}, 2×GRU({cfg['model']['hidden']}), conditioning = **{cfg['model']['cond_mode']}**"} |
| Input | 16 kHz mono → STFT (512-pt sqrt-Hann, hop 256): compressed magnitude^0.3 + compressed real/imag |
| Output | tanh-bounded complex ratio mask × noisy STFT → iSTFT overlap-add |
| Training data | LibriSpeech train-clean-100 subset ⊛ SLR28 simulated RIRs + SLR28 point-source noises, on the fly; conditions {cfg['data']['conditions']}, SNR {cfg['data']['snr_range']} dB, clipping p={cfg['data']['clip_prob']} |
| Loss | {cfg['loss']['w_sisdr']}·(−SI-SDR) + {cfg['loss']['w_stft']}·multi-resolution STFT + {cfg['loss']['w_aux']}·degradation-estimation (Smooth-L1) |
| Latency | 32 ms algorithmic (window) + compute (see table) + audio device |
| Requirements | CPU only for inference; PyTorch ≥ 2.1 |
""")
    if os.path.exists("results/latency.csv"):
        st.subheader("Measured inference cost (Apple M2, 1 CPU thread)")
        st.dataframe(pd.read_csv("results/latency.csv").set_index("system").round(4), width="stretch")
    h = f"runs/{name}/history.csv"
    if os.path.exists(h):
        st.subheader("Training curves")
        hist = pd.read_csv(h)
        st.line_chart(hist.set_index(hist.epoch + 1)[["train_loss", "val_loss"]])


PAGES = {"Home": page_home, "File enhancement": page_file, "Live enhancement": page_live,
         "Analysis": page_analysis, "Model information": page_model}
with st.sidebar:
    st.markdown("### 🎙️ Speech Enhancement")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    if not available_models():
        st.error("No trained model found in runs/. Train one first: python scripts/train.py --exp E_dacrn_film")
PAGES[page]()
