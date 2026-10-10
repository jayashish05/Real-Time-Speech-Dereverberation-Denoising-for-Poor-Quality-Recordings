# Experiments and results

All numbers on this page are **measured** by `scripts/evaluate.py`, `scripts/benchmark.py` and `scripts/make_figures.py`. They are reproduced from `results/tables/*.md` (regenerate with `python scripts/make_figures.py`). Nothing is hand‑entered. Figures are in `results/figures/`.

## 1. Protocol

* **Test data:** the same 100 LibriSpeech test‑clean utterances (40 unseen speakers) under 9 conditions (see `DATASET.md` §5). Every system sees identical inputs, so comparisons are paired.
* **Metrics:** PESQ‑WB (P.862.2), STOI, SI‑SDR and SNR against the time‑aligned direct‑path target; SNR improvement relative to the unprocessed input; DNSMOS P.835 OVRL (non‑intrusive); WER from `wav2vec2‑base‑960h` greedy decoding (corpus‑level: total errors / total words).
* **Significance:** paired bootstrap (2,000 resamples) over the 400 utterance pairs of the four degraded matched sets. A difference is marked * when its 95 % CI excludes 0.
* **Training budget:** identical for every model: 6 epochs × 500 steps × batch 16 × 4 s (≈ 53 h of freshly simulated audio), AdamW 1e‑3, best checkpoint chosen by validation loss. One seed per configuration; see §7 on variance.
* **Hardware:** Apple M2 (MPS) for training; inference cost measured on one CPU thread.

## 2. Systems

| ID | System | Why it is included |
|---|---|---|
| – | Unprocessed | Lower reference |
| – | Wiener (decision‑directed, minimum‑statistics noise PSD) | Classical noise reduction. Parameters tuned on the **validation** set; flat optimum (PESQ 1.31–1.35 vs 1.29 unprocessed) |
| – | WPE (single‑channel, nara_wpe, 10 taps, delay 3, 3 iterations) | Classical dereverberation |
| – | WPE → Wiener | Classical joint system |
| A | NSNet2‑style GRU magnitude mask (2.78 M params) | Established lightweight neural real‑time baseline (DNS challenge) |
| B | CRN backbone, trained on clean + **noise only** | Spec experiment "baseline + noise augmentation" |
| C | CRN backbone, trained on clean + **reverb only** | Spec experiment "baseline + reverb augmentation" |
| D | CRN backbone, joint noise + reverb data | Spec experiment "baseline + noise + reverb"; reference for the contribution |
| D2 | D with **SI‑SDR loss only** (no multi‑resolution STFT) | Loss ablation |
| E2 | D + degradation estimator + auxiliary loss, **no FiLM** | Isolates multi‑task regularisation |
| **E** | **DA‑CRN: estimator + FiLM conditioning (proposed)** | Spec experiment "proposed adaptive system" |
| E4 | E re‑trained with the identical config | The E run was hit by 3 macOS GPU‑watchdog errors and two resume restarts; E4 is an undisturbed run and doubles as a variance check |
| E3 | FiLM driven by **oracle** labels | Upper bound for the conditioning mechanism (not deployable) |

## 3. Main results

### 3.1 Noise + reverberation test set (main condition; mean SNR 8.5 dB, T60 0.53 s)

| System | PESQ | STOI | SI‑SDR | SNRi (dB) | WER % | DNSMOS | Params | RTF (1 thread) | ms / 16 ms frame |
|---|---|---|---|---|---|---|---|---|---|
| Unprocessed | 1.277 | 0.782 | −0.46 | 0.00 | 25.7 | 1.69 | – | – | – |
| Wiener | 1.300 | 0.771 | −0.10 | 0.78 | 25.2 | 1.87 | – | 0.001 | – |
| WPE | 1.292 | 0.788 | −0.26 | 0.26 | 23.8 | 1.74 | – | 0.005 | – |
| WPE → Wiener | 1.315 | 0.778 | 0.02 | 0.88 | 23.8 | 1.93 | – | 0.005 | – |
| A NSNet2‑style | 1.392 | 0.792 | 0.05 | 2.47 | 25.4 | 2.05 | 2.78 M | 0.004 | 0.23 |
| D CRN (joint data) | 1.448 | 0.803 | 1.06 | 3.36 | 23.2 | 2.26 | 1.30 M | 0.009 | 0.53 |
| **E DA‑CRN (proposed)** | **1.456** | 0.801 | 1.09 | **3.61** | 24.4 | **2.31** | 1.48 M | 0.010 | 0.60 |
| E4 DA‑CRN (clean re‑run) | 1.473 | 0.810 | **1.20** | 3.47 | 22.5 | 2.25 | 1.48 M | 0.010 | 0.58 |
| E2 multitask, no FiLM | **1.483** | **0.811** | 1.11 | 3.51 | **20.9** | 2.30 | 1.42 M | 0.010 | 0.54 |
| E3 oracle labels | 1.464 | 0.809 | 0.94 | 3.77 | 21.4 | 2.26 | 1.48 M | 0.010 | – |
| Target (upper bound) | 4.644 | 1.000 | – | – | 3.0 | 3.22 | – | – | – |

### 3.2 PESQ‑WB by condition (`results/tables/pesq_by_condition.md`, fig07)

| System | clean | noise | reverb | noise+reverb | severe | unseen noise | unseen room | low SNR | phone |
|---|---|---|---|---|---|---|---|---|---|
| Unprocessed | 4.644 | 1.542 | 1.969 | 1.277 | 1.097 | 1.477 | 1.199 | 1.082 | 1.379 |
| Wiener | 4.216 | 1.602 | 1.917 | 1.300 | 1.107 | 1.540 | 1.220 | 1.079 | 1.398 |
| WPE | 4.445 | 1.555 | **2.105** | 1.292 | 1.116 | 1.536 | 1.216 | 1.081 | 1.391 |
| WPE → Wiener | 4.148 | 1.601 | 2.052 | 1.315 | 1.127 | **1.601** | 1.234 | 1.076 | **1.413** |
| NSNet2‑style | 4.528 | 1.711 | 1.991 | 1.392 | 1.157 | 1.551 | 1.262 | **1.147** | 1.379 |
| CRN (joint) | 4.589 | 1.790 | 2.079 | 1.448 | 1.178 | 1.596 | **1.285** | 1.134 | 1.370 |
| DA‑CRN (proposed) | 4.509 | **1.850** | 2.071 | **1.456** | **1.181** | 1.580 | 1.284 | 1.133 | 1.388 |

### 3.3 STOI, SI‑SDR, WER, SNR improvement, DNSMOS by condition

| System | STOI noise / reverb / n+r / severe | SI‑SDR noise / reverb / n+r / severe | WER noise / reverb / n+r / severe | DNSMOS n+r / severe |
|---|---|---|---|---|
| Unprocessed | 0.867 / 0.874 / 0.782 / 0.667 | 6.3 / 2.2 / −0.5 / −4.9 | 20.6 / 7.0 / 25.7 / 56.9 | 1.69 / 1.32 |
| WPE → Wiener | 0.864 / 0.875 / 0.778 / 0.656 | 7.4 / 2.4 / 0.0 / −4.0 | 15.9 / 6.4 / 23.8 / 53.8 | 1.93 / 1.49 |
| NSNet2‑style | 0.872 / 0.873 / 0.792 / 0.682 | 8.4 / 2.1 / 0.1 / −3.2 | 16.1 / 7.6 / 25.4 / 52.6 | 2.05 / 1.62 |
| CRN (joint) | 0.881 / 0.884 / 0.803 / 0.694 | 10.5 / 2.4 / 1.1 / −1.8 | 14.6 / 10.6 / 23.2 / 57.0 | 2.26 / 1.82 |
| DA‑CRN (proposed) | 0.880 / 0.884 / 0.801 / 0.686 | 10.8 / 2.3 / 1.1 / −1.6 | 15.3 / 10.1 / 24.4 / 57.0 | 2.31 / 1.90 |

Full tables: `results/tables/{stoi,si_sdr,wer,snr_improvement,dnsmos_ovrl}_by_condition.md`. Plots: fig08–fig10, fig12.

**Reading the main results**

1. **Neural vs. classical:** on every degraded matched condition the CRN‑family models give the largest PESQ, STOI, SI‑SDR, SNR‑improvement and DNSMOS gains. DA‑CRN vs. the best classical system (WPE→Wiener), pooled over the 4 degraded sets: PESQ **+0.116\***, STOI **+0.019\***, SI‑SDR **+1.71 dB\***.
2. **Noise is handled well; reverberation much less.** On the noise‑only set DA‑CRN raises PESQ 1.54 → 1.85, SI‑SDR 6.3 → 10.8 dB and cuts WER 20.6 → 15.3 %. On the reverb‑only set the gain is small (PESQ 1.97 → 2.07, SI‑SDR +0.1 dB) and classical WPE is as good (PESQ 2.105).
3. **ASR is not reliably improved.** Pooled over degraded sets, DA‑CRN vs. unprocessed WER changes by −0.82 points (95 % CI [−2.38, +0.75], not significant). WER improves strongly for additive noise but **gets worse for reverb‑only input** (7.0 → 10.1 %), for real rooms and for the phone channel. WPE never hurts ASR. This is the familiar observation that enhancement artefacts can hurt a robust ASR model even when perceptual metrics improve.
4. **Clean input is almost untouched:** PESQ 4.51–4.59 vs. 4.64; WER 2.9 % unchanged. The classical systems degrade clean speech more (Wiener 4.22).

## 4. Ablation study (fig15, fig16; `ablation*.md`)

Mean over the four degraded matched sets (noise, reverb, noise + reverb, severe):

| System | PESQ | STOI | SI‑SDR | SNRi | WER % | DNSMOS |
|---|---|---|---|---|---|---|
| B CRN, noise data only | 1.608 | 0.815 | 2.84 | 2.62 | 24.1 | 2.21 |
| C CRN, reverb data only | 1.524 | 0.805 | 0.85 | 1.28 | 26.8 | 2.04 |
| D CRN, joint data | 1.624 | 0.816 | 3.03 | 3.91 | 26.4 | 2.36 |
| D2 CRN joint, SI‑SDR loss only | 1.547 | 0.805 | 2.17 | 1.99 | 26.1 | 2.09 |
| E2 + estimator, no FiLM | 1.663 | 0.825 | 3.09 | 3.98 | 23.7 | 2.35 |
| **E DA‑CRN (proposed)** | 1.639 | 0.813 | 3.17 | **4.16** | 26.7 | **2.39** |
| E4 DA‑CRN clean re‑run | 1.669 | 0.825 | **3.31** | 4.15 | 24.2 | 2.33 |
| E3 oracle conditioning | **1.677** | 0.823 | 3.13 | 3.29 | 24.0 | 2.33 |

Per condition (PESQ, `ablation_pesq_by_condition.md`):

| System | noise | reverb | noise+reverb | severe |
|---|---|---|---|---|
| Unprocessed | 1.542 | 1.969 | 1.277 | 1.097 |
| B noise data only | **1.866** | 1.949 | 1.432 | 1.184 |
| C reverb data only | 1.547 | 2.101 | 1.306 | 1.142 |
| D joint data | 1.790 | 2.079 | 1.448 | 1.178 |
| E DA‑CRN | 1.850 | 2.071 | 1.456 | 1.181 |
| E4 DA‑CRN re‑run | 1.852 | 2.146 | 1.473 | **1.204** |
| E3 oracle | **1.866** | **2.178** | 1.464 | 1.202 |

### Paired significance (`results/tables/significance.md`)

| Comparison (a − b) | PESQ | STOI | SI‑SDR (dB) | WER (points) |
|---|---|---|---|---|
| E DA‑CRN − D CRN | +0.016 [+0.006, +0.025] * | −0.003 [−0.005, −0.001] * | +0.15 [+0.05, +0.25] * | +0.40 [−0.79, +1.70] |
| E4 DA‑CRN re‑run − D CRN | +0.045 [+0.036, +0.054] * | +0.009 [+0.007, +0.011] * | +0.28 [+0.19, +0.39] * | −2.19 [−3.38, −1.04] * |
| **E4 − E (identical config)** | +0.029 [+0.019, +0.038] * | +0.012 [+0.010, +0.014] * | +0.14 [+0.05, +0.23] * | −2.59 [−3.73, −1.46] * |
| E2 multitask − D CRN | +0.039 [+0.029, +0.051] * | +0.009 [+0.008, +0.011] * | +0.06 [−0.03, +0.15] | −2.68 [−3.77, −1.68] * |
| E3 oracle − D CRN | +0.054 [+0.041, +0.067] * | +0.007 [+0.005, +0.010] * | +0.11 [−0.03, +0.25] | −2.39 [−3.70, −1.13] * |
| D CRN − D2 SI‑SDR only | +0.076 [+0.062, +0.091] * | +0.010 [+0.008, +0.013] * | +0.86 [+0.68, +1.04] * | +0.26 [−1.13, +1.63] |
| D joint − B noise‑only data | +0.016 [+0.002, +0.030] * | +0.001 [−0.002, +0.003] | +0.18 [+0.07, +0.31] * | +2.21 [+0.82, +3.58] * |
| D joint − C reverb‑only data | +0.100 [+0.074, +0.125] * | +0.011 [+0.007, +0.014] * | +2.18 [+1.87, +2.49] * | −0.48 [−2.01, +1.10] |
| D CRN − A NSNet2‑style | +0.061 [+0.045, +0.076] * | +0.011 [+0.009, +0.013] * | +1.21 [+1.01, +1.42] * | +0.92 [−0.48, +2.43] |
| E DA‑CRN − WPE→Wiener | +0.116 [+0.089, +0.144] * | +0.019 [+0.016, +0.023] * | +1.71 [+1.36, +2.03] * | +1.75 [+0.32, +3.11] * |
| E DA‑CRN − Unprocessed | +0.168 [+0.146, +0.191] * | +0.015 [+0.012, +0.019] * | +2.40 [+2.09, +2.71] * | −0.82 [−2.38, +0.75] |

### What the ablation shows

1. **The multi‑resolution STFT loss matters most among the design choices:** dropping it (D2) loses 0.076 PESQ, 0.86 dB SI‑SDR and 1.9 dB SNRi, even though D2 optimises SI‑SDR directly.
2. **Training data must match the degradation.** The reverb‑only model (C) is clearly worst on noise (PESQ 1.55 vs 1.79). The noise‑only model (B) is surprisingly competitive, even on the reverb set (1.95 vs 1.97 unprocessed, i.e. it neither helps nor hurts). Joint training (D) is the only model that improves both: significant over C on PESQ/STOI/SI‑SDR, and over B on PESQ/SI‑SDR. B, however, has the better WER (+2.2 points in its favour).
3. **Degradation awareness gives a small but significant quality gain:** all four estimator variants (E, E4, E2, E3) beat the backbone D on PESQ (+0.016 to +0.054) and three of them on WER (−2.2 to −2.7 points).
4. **The FiLM‑specific contribution is *not* supported at this budget.**
   * E2 (estimator + auxiliary loss, **no FiLM**) is as good as or better than the FiLM models. The benefit appears to come from the auxiliary degradation‑estimation task (multi‑task regularisation), not from using the estimate to modulate processing.
   * Oracle conditioning (E3) beats estimated conditioning (E) significantly, but by about as much as two identical runs differ (E4 − E). We cannot separate "better labels" from seed / run variance with one run each.
   * The run‑to‑run difference between two identically configured trainings (E vs. E4) is **as large as the method effects** (PESQ +0.029, WER −2.6 points). Single‑seed differences below ~0.03 PESQ or ~2.5 WER points should not be over‑interpreted.

## 5. Generalisation to unseen conditions (fig14; `generalisation_*.md`)

| System | PESQ: n+r (matched) | unseen noise (DEMAND) | unseen room (real RIRs) | SNR −10…−5 dB | phone channel |
|---|---|---|---|---|---|
| Unprocessed | 1.277 | 1.477 | 1.199 | 1.082 | 1.379 |
| WPE → Wiener | 1.315 | 1.601 | 1.234 | 1.076 | 1.413 |
| NSNet2‑style | 1.392 | 1.551 | 1.262 | 1.147 | 1.379 |
| CRN (joint) | 1.448 | 1.596 | 1.285 | 1.134 | 1.370 |
| DA‑CRN (proposed) | 1.456 | 1.580 | 1.284 | 1.133 | 1.388 |
| DA‑CRN clean re‑run | 1.473 | 1.628 | 1.303 | 1.157 | 1.393 |

| System | WER %: n+r | unseen noise | unseen room | low SNR | phone |
|---|---|---|---|---|---|
| Unprocessed | 25.7 | 20.3 | 39.1 | 67.9 | 58.0 |
| WPE → Wiener | 23.8 | **19.1** | **32.7** | 64.4 | **50.7** |
| CRN (joint) | 23.2 | 24.9 | 41.3 | 60.1 | 63.2 |
| DA‑CRN (proposed) | 24.4 | 24.1 | 41.2 | 59.8 | 63.4 |
| DA‑CRN clean re‑run | 22.5 | 24.5 | 38.9 | 55.0 | 56.9 |

* **Unseen noise corpus (DEMAND):** PESQ/STOI/SI‑SDR gains carry over (DA‑CRN 1.48 → 1.58, SNRi +4.1 dB), but WER gets worse (20.3 → 24.1 %).
* **Unseen real rooms:** small PESQ gain (1.20 → 1.28, SNRi +4.4 dB). WER is not improved; WPE→Wiener helps ASR more (32.7 %).
* **SNR below the training range (−10…−5 dB):** the neural models degrade gracefully and give the largest relative gains here (SNRi +10.8 dB, WER 67.9 → 59.8 %).
* **Telephone channel (8 kHz band‑limited, μ‑law; never seen in training):** essentially no gain; WER worse for most neural models. This is a clear failure case: the models were never trained on band‑limited input.
* **Input SNR / T60 sweeps** (fig17–fig20, pooled noise, noise + reverb and low‑SNR sets): the neural PESQ advantage over unprocessed and classical is largest at 0–15 dB input SNR (≈ +0.2–0.3 PESQ) and small below −5 dB. ASR benefits **only at low SNR**: below ~5 dB WER drops by 5–15 points (e.g. 48 → 33–39 % around −2.5 dB), while above ~7 dB all systems are within ±2 points of unprocessed. Across T60 the PESQ gain of the neural models is roughly constant (+0.10–0.15) from 0.2 s to 1.4 s. The non‑monotonic T60 curve reflects the SNR mix within each T60 bin, not the room alone.

## 6. Degradation estimator (fig13; `results/estimator.csv`)

Utterance‑level estimates of DA‑CRN on 600 test utterances (matched sets + unseen rooms):

* Mean absolute error: **SNR 6.0 dB, T60 0.24 s, C50 7.5 dB.** Noise‑free and anechoic inputs are mostly recognised as such (estimates concentrate near the 40 dB / 0 s bounds, with a tail down to ~15 dB SNR).
* Degradation‑type classification (thresholds calibrated on validation): **81.3 % accuracy**. Confusions are almost all noise + reverberation → noise (57 of 300), i.e. moderate reverberation hidden under noise.

| true \ predicted | clean | noise | noise + reverb | reverb |
|---|---|---|---|---|
| clean | 93 | 6 | 0 | 1 |
| noise | 5 | 88 | 7 | 0 |
| noise + reverb | 3 | 57 | 221 | 19 |
| reverb | 8 | 1 | 5 | 86 |

## 7. Efficiency and real‑time behaviour (fig11; `results/latency.csv`)

| System | Params | GMAC/s | RTF offline, 1 CPU thread | Streaming ms per 16 ms frame (mean / p95) | Streaming RTF | RTF offline, MPS |
|---|---|---|---|---|---|---|
| NSNet2‑style | 2.78 M | 0.18 | 0.0041 | 0.23 / 0.25 | 0.014 | 0.018 |
| CRN (joint) | 1.30 M | 0.29 | 0.0089 | 0.53 / 0.56 | 0.033 | 0.017 |
| **DA‑CRN** | 1.48 M | 0.30 | 0.0098 | **0.60 / 0.65** | **0.037** | 0.025 |
| WPE → Wiener | – | – | 0.0053 (offline only) | – | – | – |

* **Algorithmic latency:** 32 ms (one 512‑sample window). There is no look‑ahead.
* **Streaming correctness:** frame‑by‑frame output equals whole‑file output (max abs. difference < 1e‑4; `tests/test_streaming.py`, 3 block sizes).
* **Live audio test** (`scripts/realtime.py`, MacBook Air built‑in mic, 16 ms blocks, run while the GPU was training): 1.3 ms compute per frame (RTF 0.08), **0 late blocks**, measured end‑to‑end latency ≈ 113 ms. This is 32 ms algorithmic + compute + CoreAudio input/output buffers reported by PortAudio. Bluetooth headsets add their own codec latency.
* Model size: 5.7 MB (fp32). Peak process memory during benchmarking: 0.69 GB, which includes PyTorch and the benchmark harness.

## 8. Overall conclusions

1. A compact (1.5 M parameter) causal CRN trained on simulated noise + reverberation gives **significant, real‑time** improvements in perceptual quality: on the joint condition PESQ 1.28 → 1.46, DNSMOS 1.69 → 2.31, SI‑SDR +1.5 dB, at 0.6 ms of compute per 16 ms frame. It clearly beats classical WPE/Wiener processing on these metrics.
2. **Noise is the solved part; reverberation is not.** Dereverberation gains are small, and processing reverberant speech hurts downstream ASR. Classical WPE remains the better choice when ASR on reverberant speech is the goal.
3. **Degradation awareness:** adding a supervised degradation estimator improves quality slightly but significantly. Feeding its output back through FiLM does **not** add a measurable benefit beyond the auxiliary‑task effect, and seed variance is of the same size as these effects. The estimator itself works well enough (81 % type accuracy, 6 dB SNR MAE) to drive the user‑facing degradation report.
4. To strengthen these conclusions, the next steps would be: several seeds per configuration, longer training (all models were still improving at epoch 6), early‑reflection training targets for dereverberation, and band‑limited / codec augmentation for phone‑quality input.
