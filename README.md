# Real‑Time Speech Dereverberation and Denoising for Poor‑Quality Recordings

An end‑to‑end, research‑style speech enhancement system that **jointly removes background noise and room reverberation** (and tolerates clipped recordings). It runs **causally in real time on a laptop CPU**: microphone in, enhanced speech out, 32 ms algorithmic latency.

The proposed model, **DA‑CRN (degradation‑aware convolutional recurrent network)**, estimates the acoustic condition online (SNR, T60, C50 every 16 ms) and uses the estimate to adapt its own processing through FiLM conditioning. It also reports to the user *what* degradation it detected.

```
clean speech ─► room (RIR) ─► + noise ─► poor-quality speech ─► degradation analysis ─► adaptive enhancement ─► enhanced speech
                                                                  (SNR, T60, C50)         (FiLM-conditioned CRN)
```

**Start the demo:**

```bash
streamlit run app/app.py
```

| Document | Content |
|---|---|
| [DATASET.md](DATASET.md) | Data, splits, download, preprocessing, test conditions |
| [docs/dataset_comparison.md](docs/dataset_comparison.md) | DNS / WHAMR! / REVERB / VoiceBank+DEMAND / LibriSpeech / VCTK / RIR datasets compared |
| [docs/literature_review.md](docs/literature_review.md) | 39 verified references, with how this work differs |
| [docs/architecture.md](docs/architecture.md) | Model choice, DA‑CRN design, losses, complexity |
| [docs/novelty.md](docs/novelty.md) | Existing work vs. our implementation vs. our contribution |
| [docs/experiments.md](docs/experiments.md) | Baselines, ablations, generalisation, all results |
| [docs/reproducibility.md](docs/reproducibility.md) | Seeds, exact commands, artefacts |

## What is in this repository

* **Included:** all source code, configuration, tests, the trained checkpoints of all nine models (`runs/*/best.pt`, 5 to 11 MB each), training logs, every result table and figure, the per-utterance results, and the project report (`report/Project_Report.pdf`).
* **Not included:** the datasets (about 5 GB) and the generated test audio. The app, file upload and live microphone mode work straight after installation because the checkpoints are included. To use the "Example from the test set" option or to re-run training and evaluation, download and prepare the data first (Section 2).

## 1. Installation

Python 3.10–3.12. macOS, Linux and Windows are supported (tested on macOS / Apple M2).

```bash
cd speech-enhancement
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate speech-enhancement
pytest                                                  # 12 tests: STFT, degradation, streaming == offline
```

The ASR model for WER (`facebook/wav2vec2-base-960h`, ~360 MB) downloads automatically on first use. The DNSMOS model is included in `pretrained/dnsmos/` (Microsoft, CC BY 4.0).

## 2. Dataset preparation

```bash
python scripts/download_data.py        # ~3 GB: LibriSpeech subset, SLR28 RIRs + noises, DEMAND (resumable)
python scripts/prepare_data.py         # metadata, RIR acoustics (T60/C50/DRR), fixed val + 9 test sets (~1 min)
```

## 3. Training

```bash
python scripts/train.py --exp E_dacrn_film            # proposed model (~50 min on Apple M2 / MPS; CUDA used if present)
python scripts/train.py --all                          # all 8 experiments of the study
python scripts/train.py --exp E_dacrn_film --resume   # resume an interrupted run
python scripts/train.py --exp E_dacrn_film --demo     # tiny CPU-only configuration (~2 min) to check the pipeline
```

Configuration: `config/config.yaml` (shared settings + one block per experiment). Every run writes `runs/<exp>/{config.yaml,train.log,history.csv,best.pt,last.pt}`. Training includes seeding, on‑the‑fly degradation, validation, ReduceLROnPlateau, early stopping, best‑model saving and resume.

## 4. Evaluation

```bash
python scripts/evaluate.py             # PESQ, STOI, SI-SDR, SNR improvement, DNSMOS, WER for all systems × 9 test sets
python scripts/benchmark.py            # params, GMAC/s, RTF (offline and streaming), latency, memory, estimator accuracy
python scripts/make_figures.py         # results/tables/*.md and results/figures/*.png
```

## 5. Inference and real‑time use

```bash
python scripts/realtime.py                         # microphone -> DA-CRN -> speakers (use headphones), live stats
python scripts/realtime.py --list-devices
python scripts/realtime.py --no-play --seconds 15 --save results/live
python scripts/realtime.py --benchmark             # streaming latency of every model, no audio device needed
```

From Python:

```python
from src.inference.enhancer import load_model, load_audio, enhance, analyse
model = load_model("E_dacrn_film")
x = load_audio("recording.mp3")            # any rate / channels -> 16 kHz mono
y, est = enhance(model, x)                 # enhanced signal + per-frame degradation estimates
print(analyse(est))                        # {'snr_db':…, 't60_s':…, 'c50_db':…, 'degradation': 'noise + reverberation', …}
```

### The application

`streamlit run app/app.py` opens five pages:

* **Home:** project summary and system architecture.
* **File enhancement:** upload WAV/MP3/FLAC/OGG, or pick a test‑set example (which has a clean reference). Shows original and enhanced players, waveform and spectrogram comparison, degradation analysis, metrics (PESQ / STOI / SI‑SDR / SNR improvement with a reference; DNSMOS and ASR transcript always; WER if a transcript is given) and a download button.
* **Live enhancement:** (1) a true real‑time loop with the local microphone and speakers (16 ms blocks inside the audio callback), with live compute time, RTF, end‑to‑end latency, late‑block counter, live degradation estimate and A/B bypass; (2) browser recording processed by the same frame‑by‑frame streaming engine.
* **Analysis:** degradation trace and metrics for the last input, plus the test‑set result tables and figures.
* **Model information:** architecture, parameters, training data, loss, measured inference cost, training curves.

## 6. Hardware requirements

| Task | Requirement |
|---|---|
| Inference / live demo | Any recent CPU; one thread is enough (see measured RTF below). ~1 GB RAM including the app |
| Training | Apple Silicon (MPS) or an NVIDIA GPU with ≥ 4 GB; ~50 min per experiment on an M2. CPU‑only training works (`--demo`) but a full run would take several hours |
| Evaluation | CPU; ~1 h for all systems (WER via wav2vec2 dominates) |
| Disk | ~6 GB for data + ~1 GB for test sets and checkpoints |

## 7. Results (measured; full tables in [docs/experiments.md](docs/experiments.md))

Noise + reverberation test set (100 unseen‑speaker utterances, mean SNR 8.5 dB, T60 0.53 s):

| System | PESQ‑WB | STOI | SI‑SDR (dB) | SNRi (dB) | WER % | DNSMOS OVRL | Params | ms per 16 ms frame (1 CPU thread) |
|---|---|---|---|---|---|---|---|---|
| Unprocessed | 1.277 | 0.782 | −0.46 | 0.00 | 25.7 | 1.69 | – | – |
| WPE → Wiener (classical) | 1.315 | 0.778 | 0.02 | 0.88 | 23.8 | 1.93 | – | offline only |
| NSNet2‑style (neural baseline) | 1.392 | 0.792 | 0.05 | 2.47 | 25.4 | 2.05 | 2.78 M | 0.23 |
| CRN backbone | 1.448 | 0.803 | 1.06 | 3.36 | 23.2 | 2.26 | 1.30 M | 0.53 |
| **DA‑CRN (proposed)** | **1.456** | 0.801 | 1.09 | **3.61** | 24.4 | **2.31** | 1.48 M | 0.60 |

Key findings (paired bootstrap, 95 % CI, 400 degraded test utterances):

* DA‑CRN significantly improves PESQ (+0.17), STOI (+0.015) and SI‑SDR (+2.4 dB) over the input, and beats classical WPE→Wiener on all three. It runs at **RTF 0.037 frame by frame on one CPU thread with 32 ms algorithmic latency**.
* Additive noise is handled well (noise‑only set: PESQ 1.54 → 1.85, WER 20.6 → 15.3 %). **Dereverberation gains are small**, and processing reverberant speech **raises ASR WER** (reverb‑only 7.0 → 10.1 %), where classical WPE helps instead.
* The multi‑resolution STFT loss and joint noise + reverb training data are the two choices with the largest measured effect.
* Adding the degradation estimator gives a small significant gain over the plain CRN (PESQ +0.02…+0.05). FiLM conditioning on the estimate gives **no measurable benefit beyond that multi‑task effect**, and run‑to‑run variance is of the same size. Reported as measured; see [docs/novelty.md](docs/novelty.md).
* The degradation report identifies clean / noise / reverberation / both with 81 % accuracy (SNR MAE 6 dB, T60 MAE 0.24 s).

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `PortAudioError` / no devices in *Live enhancement* | Linux: `sudo apt install libportaudio2` (or the conda env). macOS: allow microphone access for the terminal/IDE in System Settings → Privacy |
| Howling/feedback in live mode | Use headphones, or `--no-play` |
| Bluetooth headset adds ~100–200 ms | Bluetooth codec latency is outside the model. Use wired or built‑in audio for latency measurements |
| `Impacting Interactivity` GPU errors on macOS | macOS kills long GPU command buffers when the GPU is shared. Train one model at a time (the trainer skips non‑finite steps) |
| Download stalls | `scripts/download_data.py` is resumable; just rerun it |
| `No trained model found` in the app | Train first, or copy `runs/E_dacrn_film/best.pt` from a trained machine |
| PESQ returns NaN | The reference contains no detectable speech (very short/silent clip). Such utterances are excluded from PESQ means |

## 9. Limitations

* Trained on simulated mixtures from 40 LibriSpeech speakers (15.6 h) and SLR28 rooms/noises. Real‑world robustness is tested only via held‑out real RIRs, a different noise corpus, a telephone channel and live recordings, not on a large real noisy‑reverberant corpus with references.
* 16 kHz only (wide‑band). No bandwidth extension: telephone‑band input stays band‑limited.
* The dereverberation target is the direct path. Very strong late reverberation (T60 > 1 s at low SNR) is reduced but not removed.
* Degradation estimates are approximate (see the estimator accuracy in `docs/experiments.md`) and need ~1 s of audio to settle.
* Live mode in the app uses the audio devices of the machine running Streamlit. On a remote server use the browser‑recording mode, which is not real‑time.
* Results are not directly comparable with published WHAMR!/REVERB/VoiceBank numbers (different, openly licensed test data).
