# Dataset

The comparison of candidate corpora and the reasons for this choice are in [docs/dataset_comparison.md](docs/dataset_comparison.md). In short: degraded speech is simulated from openly licensed 16 kHz corpora, with speaker‑, room‑ and noise‑disjoint splits. DNS (~1 TB), WHAMR!/REVERB (LDC licences) and VoiceBank+DEMAND (no reverberation) were rejected.

## 1. What is used

| Role | Corpus | Subset actually used | License |
|---|---|---|---|
| Clean speech: train | LibriSpeech `train-clean-100` | first 1.0 GB of the archive → **40 speakers, 4,322 utterances, 15.6 h** | CC BY 4.0 |
| Clean speech: validation | LibriSpeech `dev-clean` | 40 speakers, 2,703 utterances, 5.4 h | CC BY 4.0 |
| Clean speech: test | LibriSpeech `test-clean` | 40 speakers, 2,620 utterances, 5.4 h (100 utterances of 3–10 s sampled for the test sets) | CC BY 4.0 |
| RIRs: train / val / test | OpenSLR SLR28 `simulated_rirs` | 20 RIRs per room; rooms 1–160 train, 161–180 val, 181–200 test; T60 ≤ 1.5 s → **6,438 / 807 / 785 RIRs** | Apache 2.0 |
| RIRs: unseen rooms | SLR28 `real_rirs_isotropic_noises` (RWCP, REVERB 2014, Aachen AIR; channel 0) | 227 measured RIRs with 0.05 < T60 ≤ 1.5 s | Apache 2.0 (see SLR28 README for source terms) |
| Noise: train / val / test | SLR28 `pointsource_noises` (from MUSAN) | split by CRC32 of file name → **672 files (4.8 h) / 75 (0.6 h) / 79 (0.5 h)** | Apache 2.0 / per‑file MUSAN terms (`pointsource_noises/LICENSE`) |
| Noise: unseen corpus | DEMAND 16 kHz, channel 1 | DLIVING, OOFFICE, PCAFETER, TBUS (4 × 5 min) | CC BY 4.0 |

Speaker disjointness is asserted in `scripts/prepare_data.py`. LibriSpeech's official splits share no speakers.

**Deviation from the plan:** the fifth DEMAND test environment (SPSQUARE) could not be downloaded; the transfer stalled repeatedly on the network used. The unseen‑noise test therefore uses 4 environments.

## 2. Download

```bash
python scripts/download_data.py            # ~3.0 GB transferred, resumable; --train-gb sets the train subset size
```

* LibriSpeech dev/test: full archives (337 + 346 MB).
* LibriSpeech train‑clean‑100: only a **prefix of the archive** (default 1.0 GB). Members are stored speaker by speaker, so this yields complete speakers. The member cut by the truncation is deleted.
* SLR28: only the needed members are fetched by **HTTP range requests** into a sparse copy of the zip (simulated RIRs, real RIRs, point‑source noises ≈ 793 MB; the 0.5 GB of isotropic noises is skipped).
* DEMAND: per‑environment 16 kHz zips from Zenodo record 1227121.

Disk after extraction: LibriSpeech 1.7 GB, SLR28 subset 2.9 GB (simulated RIRs are stored uncompressed), DEMAND 37 MB (channel 1 only), generated val/test sets 0.8 GB. **Total ≈ 5.5 GB.**

## 3. Pre‑processing (`scripts/prepare_data.py`, seed 1234)

1. **Resampling / format:** all corpora are 16 kHz mono already. Multichannel files use channel 0 (RIRs) or channel 1 (DEMAND). User uploads in the app are converted to mono and resampled with `scipy.signal.resample_poly`.
2. **RIR analysis:** for every RIR, T60 (Schroeder backward integration, T20 fit −5…−25 dB extrapolated to −60 dB, after noise‑floor compensation of measured RIRs), C50 and DRR are computed (`src/preprocessing/dsp.py:rir_params`). RIRs with T60 ≤ 0.05 s (failed estimates) or > 1.5 s are discarded. The SLR28 large rooms reach ~7 s, which is unrealistic for speech recordings.
3. **RIR normalisation:** pre‑delay is cut 2.5 ms before the direct‑path peak, and the RIR is scaled so the direct path (±2.5 ms) has unit energy. The **target** is the clean speech convolved with this direct path only. It is time‑aligned with the reverberant mixture.
4. **Mixing** (`src/preprocessing/dsp.py:degrade`): clean → ⊛ RIR → + noise scaled to the drawn SNR **relative to the reverberant speech** → random level −35…−15 dBFS (same gain on the target) → optional clipping at 30–80 % of the peak (training only, p = 0.1) → peak‑safe rescale.
5. **Chunking:** training uses random 4 s crops (shorter utterances are zero‑padded; noise is tiled if shorter). Validation uses 6 s crops. Test sets use full 3–10 s utterances.
6. **Normalisation at inference:** inputs are scaled to −25 dBFS (middle of the training level range) and the output is scaled back.

## 4. Training data (on the fly)

Every training example is simulated on the fly with `rng = default_rng([seed, epoch, index])`: reproducible, but never repeated across epochs. Default condition mix: clean 10 %, noise 30 %, reverb 20 %, noise + reverb 40 %. SNR ~ U(−5, 20) dB. Experiments B and C change the mix (`config/config.yaml`).

## 5. Fixed validation and test sets

* **Validation:** 256 mixtures of 6 s from dev‑clean with val rooms and val noises, using the training condition mix (`data/metadata/val.csv`).
* **Test:** the **same 100 test‑clean utterances** in each of 9 conditions (paired comparison). Every parameter (noise file, RIR, SNR, T60, C50, DRR, clipping) is stored in `data/metadata/test.csv`.

| Test set | Noise | Room | SNR (dB) | Purpose |
|---|---|---|---|---|
| `clean` | – | – | – | Does processing damage clean speech? |
| `noise` | SLR28 test noises | – | U(−5, 20) | Denoising only |
| `reverb` | – | SLR28 test rooms | – | Dereverberation only |
| `noise_reverb` | SLR28 test noises | SLR28 test rooms | U(−5, 20) | Joint task (main condition) |
| `severe` | SLR28 test noises | test rooms with T60 ≥ 75th percentile (mean T60 1.03 s) | U(−5, 5) | Severe noise + severe reverb |
| `unseen_noise` | **DEMAND** (other corpus) | SLR28 test rooms | U(−5, 20) | Noise‑type generalisation |
| `unseen_room` | SLR28 test noises | **real measured RIRs** | U(−5, 20) | Simulated‑to‑real room generalisation |
| `unseen_low_snr` | SLR28 test noises | SLR28 test rooms | **U(−10, −5)**, below the training range | SNR extrapolation |
| `phone` | SLR28 test noises | SLR28 test rooms | U(−5, 20), then **8 kHz, 300–3400 Hz band‑pass, μ‑law** | Mobile/telephone‑quality channel, never seen in training |

Mean conditions per set (from `test.csv`): noise 6.3 dB SNR; reverb T60 0.52 s, C50 15.5 dB; severe −0.4 dB SNR, T60 1.03 s, C50 10.3 dB; unseen‑low‑SNR −7.8 dB.

## 6. Leakage checks

* Speakers: train ∩ test = ∅ (asserted); val speakers are a third disjoint set.
* Rooms: simulated rooms are split by room ID, so no room appears in two splits. Real RIRs appear only in `unseen_room`.
* Noise: point‑source noise files are split by file. DEMAND is used only in `unseen_noise`.
* Model selection (early stopping, LR schedule, Wiener tuning) uses only the validation set.
