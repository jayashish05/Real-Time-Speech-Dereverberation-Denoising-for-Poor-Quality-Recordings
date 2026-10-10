# Reproducibility

## Determinism

| Component | How it is made reproducible |
|---|---|
| Data download | Fixed URLs and fixed byte prefix of `train-clean-100` (`--train-gb 1.0`) → the same speaker subset every time |
| Metadata and splits | `scripts/prepare_data.py --seed 1234`. RIR sub‑sampling uses the seeded RNG; noise splits use CRC32 of the file name (seed‑independent); room splits use the room number |
| Validation / test mixtures | Generated once by `prepare_data.py` with the same RNG and stored as float WAV in `data/processed/`, with every parameter in `data/metadata/{val,test}.csv` |
| Training mixtures | Generated on the fly from `np.random.default_rng([seed, epoch, index])`, so item *i* of epoch *e* is identical across runs, independent of DataLoader worker scheduling |
| Model init / optimiser | `seed: 42` in `config/config.yaml` (`random`, `numpy`, `torch` seeded) |
| Remaining non‑determinism | GPU/MPS kernels are not bit‑deterministic; reruns can differ in the last digits of metrics |

## Full pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py                 # ~3 GB, resumable
python scripts/prepare_data.py                  # metadata + fixed val/test sets (~1 min)
pytest                                          # STFT, degradation and streaming-equivalence tests
python scripts/train.py --all                   # 8 experiments, 6 epochs each (~50 min each on an Apple M2, MPS)
python scripts/benchmark.py                     # latency / RTF / params / estimator accuracy
python scripts/evaluate.py                      # all metrics, all systems, all test sets (cached, resumable)
python scripts/make_figures.py                  # tables in results/tables, figures in results/figures
streamlit run app/app.py                        # demo application
```

## Artefacts

| Path | Content |
|---|---|
| `runs/<exp>/config.yaml` | Fully merged config actually used by the run |
| `runs/<exp>/train.log`, `history.csv` | Step and epoch logs (train/val losses, SI‑SDR, LR, time) |
| `runs/<exp>/best.pt`, `last.pt` | Best‑validation weights; full resume state (model, optimiser, scheduler, early‑stopping counters) |
| `results/per_utterance.csv` | One row per (system, test utterance) with every metric |
| `results/latency.csv`, `results/estimator.csv` | Cost measurements; degradation‑estimator predictions |
| `results/tables/*.md`, `results/figures/*.png` | Report tables and figures, all derived from the CSVs above |

## Environment used for the reported numbers

Apple M2 (8‑core CPU, 10‑core GPU), 16 GB RAM, macOS, Python 3.12.8, PyTorch 2.14.1 (MPS backend for training, CPU for inference and latency). Exact package versions: `requirements.txt`.
