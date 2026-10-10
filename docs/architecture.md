# Architecture

## 1. Selection rationale

Hard constraints: (i) causal, frame‑by‑frame streaming on a laptop CPU; (ii) trainable from scratch on an Apple M2 (no CUDA) in about an hour per experiment; (iii) handles **both** noise and reverberation; (iv) can host a degradation‑aware mechanism.

| Candidate | Causal | CPU real‑time | Trainable here | Handles reverb well | Verdict |
|---|---|---|---|---|---|
| SEGAN, MetricGAN+, MP‑SENet, CMGAN | no | – | heavy / unstable | – | rejected: not streaming |
| Conv‑TasNet / DPRNN (time domain) | causal variant is weak / chunked | yes | yes | fragile under reverb | rejected |
| Demucs (waveform, real‑time) | yes | yes | large model, slow on MPS | denoising‑oriented | rejected for size |
| FullSubNet | look‑ahead | heavy | slow | ok | rejected for compute |
| DeepFilterNet 2/3 | yes | yes (RTF 0.19 on one notebook thread) | needs DNS‑scale data and a Rust pipeline | yes (deep filter) | strong reference; too complex to retrain here |
| GTCRN | yes | yes (very light) | yes | limited capacity | good but too small for dereverberation |
| **CRN with complex mask (CRN + DCCRN‑E masking)** | **yes** | **yes** | **yes (≈0.8 s/step on MPS incl. loss)** | **yes (phase‑aware mask)** | **selected as backbone** |

The CRN family is the established real‑time design (CRN, DCCRN, CRUSE, GTCRN). We use real‑valued convolutions on stacked real/imag/magnitude channels, which is cheaper than DCCRN's complex layers. The complex ratio mask keeps the ability to correct phase, which matters for reverberation.

## 2. DA‑CRN

```
wav (16 kHz) ─ STFT 512/256 sqrt‑Hann ─┬──────────────────────────────────────────────────────────┐
                                        │ features: |X|^0.3·cos∠X, |X|^0.3·sin∠X, |X|^0.3 (256 bins) │
                                        ▼                                                          │
                     5 × CausalConv2d (time k=2, freq k=5 stride 2) + BN + PReLU                    │
                       channels 16‑32‑48‑64‑64, freq 256→8           ──── skips ────┐              │
                                        │ z_t ∈ R^512                                │              │
                    ┌───────────────────┴──────────────┐                             │              │
                    ▼                                  │                             │              │
   Degradation estimator: GRU(64) → Linear(3)          │                             │              │
   ê_t = [SNR, T60, C50] (normalised)  ──► FiLM: z ← z·(1+γ(ê_t)) + β(ê_t)           │              │
                                        ▼                                            │              │
                        2 × GRU(256) → Linear(512)                                   │              │
                                        ▼                                            ▼              │
                     5 × ConvTranspose2d (freq) with skip concatenation → 2 channels                │
                                        ▼                                                           │
          complex mask M;  |M| ← tanh|M|  (bounded, DCCRN‑E);  Y = M ⊙ X  ◄──────────────────────────┘
                                        ▼
                               iSTFT, overlap‑add → enhanced wav
```

* **Input representation:** STFT, 512‑sample (32 ms) sqrt‑Hann window, 256 hop (16 ms), 257 bins. The Nyquist bin is dropped inside the network and reuses the last mask value. Power‑law compression (0.3) evens out the dynamic range, so the model is robust to input level.
* **Output representation:** a bounded complex ratio mask applied to the noisy STFT. Magnitude in [0, 1), free phase rotation.
* **Causality:** encoder convolutions see the current and one previous frame; the decoder works per frame; all GRUs are unidirectional. **There is no look‑ahead**: algorithmic latency is one window, 32 ms.
* **Streaming:** each encoder layer caches one input frame, and each GRU keeps its hidden state. `stream_step()` processes one frame and is verified to equal whole‑utterance inference (`tests/test_streaming.py`, max error < 1e‑4).
* **Conditioning modes** (same code, `cond_mode`): `none` (backbone), `multitask` (estimator + aux loss, no FiLM), `film` (proposed), `oracle` (FiLM on true labels, upper bound). FiLM's last layer is zero‑initialised, so training starts as the plain backbone.

### Degradation labels
Labels are the simulation parameters. **SNR** (dB, relative to reverberant speech, 40 for no noise) is clipped to [‑10, 40]. **T60** (s) is computed from each RIR by Schroeder backward integration (T20 fit, extrapolated to ‑60 dB), clipped to [0, 1.5], with 0 for no reverb. **C50** (dB, clarity: early 50 ms vs. late energy, 40 for no reverb) is clipped to [‑5, 40]. Each is normalised to roughly unit range. The estimator is trained with Smooth‑L1 on every frame against the utterance label, which forces a causal running estimate.

## 3. Training target and losses

* **Target:** the dry utterance convolved with the RIR's direct path only (±2.5 ms around the peak, unit energy). It is time‑aligned with the reverberant mixture, so dereverberation is learned without a delay‑compensation problem. For noise‑only and clean conditions the target is the clean signal.
* **Loss:** `L = 0.1·(−SI‑SDR) + 1.0·MR‑STFT + 0.5·L_aux`.
  * −SI‑SDR (Le Roux 2019) rewards waveform fidelity, including phase.
  * Multi‑resolution STFT (spectral convergence + log‑magnitude L1 at 256/512/1024‑point FFTs; Yamamoto 2020) penalises spectral artefacts and residual reverberant tails. SI‑SDR alone under‑weights these at low energy.
  * `L_aux`: degradation estimation (only for runs with an estimator).
  * The 0.1 weight puts SI‑SDR (tens of dB) on the same scale as the STFT term (≈1). Run D2 ablates the STFT term.
* **Optimisation:** AdamW, lr 1e‑3, grad‑clip 5, ReduceLROnPlateau (×0.5, patience 1), early stopping (patience 3) on validation loss, best‑model checkpointing, 6 epochs maximum (same budget for every experiment). One epoch = 500 steps × batch 16 × 4 s (≈ 8.9 h of freshly simulated audio per epoch).

## 4. Complexity and latency

See `results/latency.csv` and the table in `README.md` for measured values (parameters, GMAC/s, offline RTF, streaming ms per frame on one CPU thread, MPS RTF). Algorithmic latency is 32 ms. End‑to‑end latency with an audio device is measured live in the app and `scripts/realtime.py`.
