# Dataset comparison and selection

Written **before any data was downloaded**. Facts were checked against the official pages on 2026‑10‑05 (links below). Hardware for the project: Apple M2, 16 GB RAM, no CUDA GPU (PyTorch MPS backend), ~38 GB free disk. That budget matters for every decision below.

## 1. Candidates

| Dataset | Contents | Clean | Noisy | Reverberant | Noise recs | RIRs | fs | Speakers | Size | License / access |
|---|---|---|---|---|---|---|---|---|---|---|
| **Microsoft DNS Challenge 5** (ICASSP 2023) | LibriVox read speech, VCTK, emotional, multilingual + noise + RIRs, plus synthetic and real test sets | yes | synthesised on the fly | synthesised (RIRs from SLR26/28) | 58 GB (AudioSet, Freesound, DEMAND) | 5.9 GB | 48 kHz | thousands | ~550 GB archived, ~1 TB unpacked | CC BY 4.0 / PD / ODC mix, free |
| **WHAM! / WHAMR!** | wsj0‑2mix (2‑speaker) + real café/bar noise (WHAM!) + image‑method reverb (WHAMR!) | yes | yes | yes (simulated, anechoic targets) | 17 GB zip (35 GB unzipped) | simulated, parameters in scripts | 8/16 kHz | 101 train (WSJ0) | noise 17 GB + WSJ0 | noise CC BY‑NC 4.0; **WSJ0 speech needs a paid LDC licence** |
| **REVERB Challenge 2014** | SimData (WSJCAM0 ⊛ measured RIRs + 20 dB stationary noise), RealData (MC‑WSJ‑AV, T60 ≈ 0.7 s) | yes (SimData) | stationary, 20 dB only | yes (measured, 3 rooms × 2 distances; real room) | stationary only | measured, 8‑ch | 16 kHz | WSJCAM0 | ~ tens of GB | **LDC licence** (free evaluation licence during the challenge only) |
| **VoiceBank + DEMAND** (Valentini 2016) | VCTK subset + DEMAND/babble/speech‑shaped noise, fixed mixtures | yes | yes (train SNR 0/5/10/15, test 2.5/7.5/12.5/17.5 dB) | **no** | embedded | none | 48 kHz | 28 (or 56) train, 2 test | 2.3 + 2.6 GB (28‑spk) | Edinburgh DataShare end‑user licence (CC BY 4.0) |
| **LibriSpeech** | Read audiobook speech with transcripts | yes (mostly clean, some audiobook room colouring) | no | no | no | no | 16 kHz | train‑clean‑100: 251, dev‑clean: 40, test‑clean: 40 (disjoint) | 6.3 GB (100 h) + 337 MB + 346 MB | CC BY 4.0 |
| **VCTK 0.92** | Studio read speech, 110 speakers, ~44 h | yes (studio) | no | no | no | no | 48 kHz | 110 | 10.9 GB zip | CC BY 4.0 |
| **OpenSLR SLR28 (RIRS_NOISES)** | Simulated RIRs (small/medium/large rooms), real RIRs (RWCP, REVERB 2014, Aachen AIR), RWCP isotropic noises, point‑source noises from MUSAN | – | – | – | yes (point‑source + isotropic) | yes (sim + real) | 16 kHz | – | 1.3 GB zip | Apache 2.0 |
| **DEMAND** | 18 real environments (domestic, office, public, transport, nature, street), 16‑ch arrays, 5 min each | – | – | – | yes | – | 16 / 48 kHz | – | 16 kHz zips 78–130 MB each (~1.9 GB all) | CC BY 4.0 (Zenodo 1227121) |
| **EARS / EARS‑WHAM / EARS‑Reverb** (2024) | 107 speakers, ~100 h anechoic, expressive; benchmarks with WHAM! noise (−2.5…17.5 dB) and real RIRs | yes (anechoic) | yes | yes (real RIRs) | WHAM! | real | 48 kHz | 107 | ~100 h at 48 kHz/32‑bit (tens of GB) | CC BY‑NC 4.0 |
| **URGENT 2024/2025 challenge data** | Universal SE: noise, reverb, bandwidth limit, clipping (+codec, packet loss, wind in 2025), multiple fs | yes | simulated | simulated | many corpora | many corpora | 8–48 kHz | large | hundreds of GB | mixed per source corpus |

Sources: [DNS‑Challenge repo](https://github.com/microsoft/DNS-Challenge), [WHAM!/WHAMR!](http://wham.whisper.ai/), [REVERB summary, EURASIP JASP 2016](https://doi.org/10.1186/s13634-016-0306-6), [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791), [LibriSpeech SLR12](https://www.openslr.org/12/), [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443), [SLR28](https://www.openslr.org/28/), [DEMAND on Zenodo](https://zenodo.org/records/1227121), [EARS](https://arxiv.org/abs/2406.06185), [URGENT 2024](https://arxiv.org/abs/2406.04660), [URGENT 2025](https://arxiv.org/abs/2505.23212).

## 2. Suitability

| Dataset | Supervised training | Dereverb | Denoise | Joint | Practical here | Publishable evaluation |
|---|---|---|---|---|---|---|
| DNS 5 | ✔ | ✔ | ✔ | ✔ | ✘ 550 GB, 48 kHz, ≫ disk | ✔ (blind real test sets, DNSMOS) |
| WHAMR! | ✔ | ✔ | ✔ | ✔ | ✘ needs WSJ0 (LDC, paid) | ✔ standard benchmark |
| REVERB | ✔ (SimData) | ✔ | weak (20 dB stationary) | weak | ✘ LDC licence | ✔ standard dereverb benchmark |
| VoiceBank+DEMAND | ✔ | ✘ no reverb | ✔ | ✘ | ✔ 5 GB, but 48 kHz and 28 speakers | ✔ for denoising only |
| LibriSpeech | clean source only | via simulation | via simulation | via simulation | ✔ | ✔ with transcripts → **WER** |
| VCTK | clean source only | via simulation | via simulation | via simulation | ~ 11 GB, 48 kHz, no standard split | ✔ |
| SLR28 | RIR/noise source | ✔ (sim + **real** RIRs) | ✔ (noises) | ✔ | ✔ 1.3 GB | ✔ (used by Kaldi, DNS) |
| DEMAND | noise source | – | ✔ | – | ✔ per‑environment download | ✔ (VoiceBank test noise) |
| EARS | ✔ | ✔ | ✔ | separate benchmarks | ~ size, 48 kHz, NC licence | ✔ |
| URGENT | ✔ | ✔ | ✔ | ✔ | ✘ size | ✔ |

## 3. Decision

**Simulate noisy‑reverberant speech on the fly from three openly licensed, 16 kHz corpora:**

| Role | Corpus | Why |
|---|---|---|
| Clean speech (train / val / test) | **LibriSpeech** train‑clean‑100 / dev‑clean / test‑clean | Native 16 kHz, CC BY 4.0, **official speaker‑disjoint splits** (no speaker leakage), and **transcripts**, which make WER evaluation possible. 100 h is more speech than we can iterate over on an M2 anyway. |
| RIRs (train / val / matched test) | **SLR28 simulated RIRs**, split **by room** | Thousands of rooms of three size classes; we compute T60, C50 and DRR of every RIR ourselves (Schroeder integration), which gives the labels the degradation‑aware model needs. |
| RIRs (unseen‑room test) | **SLR28 real RIRs** (RWCP, REVERB 2014, Aachen AIR) | Measured rooms never seen in training: tests sim‑to‑real generalisation of the dereverberation. |
| Noise (train / val / matched test) | **SLR28 point‑source noises** (from MUSAN), split **by file** | Diverse, already 16 kHz, Apache 2.0, comes in the same 1.3 GB archive. |
| Noise (unseen‑noise test) | **DEMAND**, environments of the VoiceBank+DEMAND test set (DLIVING, OOFFICE, PCAFETER, TBUS; SPSQUARE was planned but its download kept stalling) | A different noise corpus, recorded with a different microphone array, for a cross‑corpus noise test. |

Download budget as planned: 6.3 + 0.34 + 0.35 GB (LibriSpeech) + 1.3 GB (SLR28) + ~0.52 GB (DEMAND) ≈ 8.8 GB. The network available ran at ~1 MB/s, so the final downloader fetches only what is used (a 1 GB speaker subset of train‑clean‑100, and only the needed SLR28 members via HTTP range requests): **≈ 3 GB transferred**. See `DATASET.md`.

**Why not the bigger benchmarks?** DNS‑5 (~1 TB) and URGENT do not fit the hardware. WHAMR! and REVERB depend on LDC corpora (WSJ0/WSJCAM0) that a student cannot freely obtain. VoiceBank+DEMAND has no reverberation, so it cannot evaluate the joint task. All of them inspired our protocol: the DNS recipe (LibriVox‑style speech, SLR28 RIRs, DEMAND/MUSAN noise, on‑the‑fly mixing) is essentially what we reproduce at small scale, and the DEMAND test environments mirror VoiceBank+DEMAND's.

**Honest limitations of the choice**
* Results are not directly comparable to published WHAMR!/REVERB/VoiceBank numbers, because our test set is our own simulation (fully documented and reproducible from the seed).
* LibriSpeech "clean" speech is audiobook audio. Some recordings carry mild room colouring or noise, so the clean reference is not anechoic.
* There are no real noisy‑reverberant recordings with clean references. For real recordings we report non‑intrusive DNSMOS plus ASR WER only.
