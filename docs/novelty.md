# Contribution: what is existing work, what is ours

We do **not** claim a new architecture family. This page separates three things.

## 1. Existing research we build on (not ours)

| Idea | Source |
|---|---|
| Causal convolutional‑recurrent encoder/decoder for real‑time SE | Tan & Wang 2018 (CRN); Braun et al. 2021 (CRUSE) |
| Complex ratio mask with tanh‑bounded magnitude | Hu et al. 2020 (DCCRN) |
| Power‑law compressed spectral input | widely used (e.g. DeepFilterNet, GTCRN) |
| SI‑SDR loss; multi‑resolution STFT loss | Le Roux et al. 2019; Yamamoto et al. 2020 |
| Telling a network about the degradation (noise‑aware training) | Seltzer et al. 2013; Xu et al. 2014 |
| Reverberation‑time‑aware dereverberation | Wu et al. 2017 (TASLP 25(1)) |
| Blind T60 / DRR estimation | ACE challenge, Eaton et al. 2016 |
| FiLM conditioning | Perez et al. 2018 |
| WPE dereverberation; decision‑directed Wiener filtering | Nakatani et al. 2010; Ephraim & Malah 1984 |
| On‑the‑fly simulation with SLR28 RIRs + noises | Ko et al. 2017; DNS challenge recipe |

## 2. Our implementation (engineering, not novelty)

* A complete, reproducible small‑scale joint denoising + dereverberation benchmark built only from openly licensed 16 kHz data. It is speaker‑, room‑ and noise‑disjoint, and every degradation parameter is logged per utterance (`data/metadata/test.csv`).
* Re‑implementations trained on the same data: the NSNet2‑style baseline, the CRN backbone, WPE and Wiener baselines.
* A streaming engine whose output is **sample‑identical to offline inference** (unit‑tested). The end‑to‑end latency and compute are measured on real audio hardware, not estimated.

## 3. Proposed contribution (what we test)

**DA‑CRN: online, multi‑factor degradation estimation that continuously conditions a single causal joint denoising + dereverberation network.**

Concretely:

1. A causal estimator branch (GRU on the encoder bottleneck) predicts **SNR, T60 and C50 every 16 ms**. It is trained with a supervised auxiliary loss on labels that come for free from the simulation (T60/C50 measured from each RIR by Schroeder integration).
2. The estimate drives **FiLM modulation** of the bottleneck before the enhancement GRU. The network can therefore change its behaviour continuously with the acoustic condition (noise‑dominated vs. reverberation‑dominated vs. both) instead of applying one fixed mapping.
3. The same estimate is exposed to the user as a **degradation report** (noise / reverberation / both, severity, live trace). This makes the system explainable and lets it state which degradation it is dealing with.

How this differs from the closest prior work:

* **Wu et al. 2017** condition on **T60 only**, for **reverberation only**, by switching discrete DNN configurations (frame shift/context) offline. We estimate **three** quantities covering **noise and reverberation jointly**, online and causally, with continuous FiLM modulation inside one streaming model.
* **Noise‑aware training (Seltzer 2013; Xu 2014)** appends a noise estimate to the input features. We estimate interpretable acoustic parameters with supervision and inject them deep in the network via FiLM. We also test whether that beats simply training the estimator as an auxiliary task.
* **Two‑stage systems (Zhao et al. 2019)** hard‑code the order "denoise, then dereverberate" with two networks. We use one network whose processing is steered by the estimated mix of degradations.

## 4. How the claim is tested (no claim without measurement)

The ablation in `docs/experiments.md` separates the effects:

| Run | What it isolates |
|---|---|
| D: CRN backbone, joint data | baseline without any degradation awareness |
| E2: + estimator, auxiliary loss, **no FiLM** | does the multi‑task signal alone help (regularisation)? |
| **E: + estimator + FiLM (proposed)** | does *using* the estimate to adapt processing help? |
| E3: FiLM with **oracle** labels | upper bound: how much an ideal estimator would give |

The contribution counts as supported only where E beats both D and E2 on the measured test sets. The gap between E and E3 tells us how much is lost to estimation error. Whatever the outcome, `docs/experiments.md` reports it as measured.

## 5. Outcome (measured, `docs/experiments.md` §4)

* **Partly supported:** degradation‑aware training helps. Every variant with the degradation estimator (E, E4, E2, E3) beats the plain backbone D on PESQ by +0.016 to +0.054 (paired 95 % CIs exclude 0), and three of the four reduce WER by 2.2–2.7 points.
* **Not supported:** the specific claim that *using* the estimate through FiLM adds value. The estimator‑without‑FiLM model E2 is as good as the FiLM models, and two identically configured FiLM runs (E vs. E4) differ by as much as the method effects. At this training budget (6 epochs, one seed), the benefit is best explained by the auxiliary degradation‑estimation task acting as a regulariser.
* **Still delivered:** the online degradation estimate drives an interpretable degradation report (81 % type accuracy, SNR MAE 6 dB), shown live in the application.
* What would test the FiLM hypothesis properly: several seeds per configuration, longer training, and a stronger estimator (the oracle model E3 shows the upper bound is only modestly higher).
