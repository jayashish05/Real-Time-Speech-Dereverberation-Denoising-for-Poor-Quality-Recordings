"""Training loop: config merging, seeding, on-the-fly data, validation, checkpointing, LR scheduling,
early stopping, best-model saving and resume."""
import copy, json, logging, os, random, time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FixedSet, OnTheFlyDataset
from src.losses.losses import EnhancementLoss, si_sdr
from src.models.dacrn import DACRN
from src.models.nsnet2 import NSNet2


def deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict) and k != "conditions":
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def load_config(exp, path="config/config.yaml", demo=False):
    base = yaml.safe_load(open(path))
    cfg = deep_update(copy.deepcopy({k: v for k, v in base.items() if k not in ("experiments", "demo")}),
                      base["experiments"][exp])
    if demo:
        deep_update(cfg, base["demo"])
    cfg["exp"] = exp
    return cfg


def build_model(cfg):
    m = cfg["model"]
    if m["name"] == "nsnet2":
        return NSNet2()
    return DACRN(channels=tuple(m["channels"]), hidden=m["hidden"], cond_mode=m["cond_mode"])


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def run_batch(model, noisy, target, lab, loss_fn):
    cond = lab if model.cond_mode == "oracle" else None
    est, est_lab = model(noisy, cond)
    loss, parts = loss_fn(est, target, est_lab, lab)
    return loss, parts, est


def train(cfg, resume=False):
    seed_all(cfg["seed"])
    tc, P = cfg["train"], cfg["paths"]
    out = os.path.join(P["runs"], cfg["exp"])
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"))
    log = logging.getLogger(cfg["exp"])
    log.setLevel(logging.INFO)
    if not log.handlers:
        for h in (logging.FileHandler(f"{out}/train.log"), logging.StreamHandler()):
            h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            log.addHandler(h)

    dev = pick_device(tc["device"])
    model = build_model(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tc["lr_factor"], patience=tc["lr_patience"])
    loss_fn = EnhancementLoss(**cfg["loss"])
    state = {"epoch": 0, "best": float("inf"), "bad": 0, "history": []}
    if resume and os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log.info(f"resumed from epoch {state['epoch']}")
    log.info(f"exp={cfg['exp']} device={dev} params={n_params:,}")

    train_ds = OnTheFlyDataset(P["speech_csv"], P["noise_csv"], P["rir_csv"], "train", cfg["data"],
                               length=tc["steps_per_epoch"] * tc["batch_size"], seed=cfg["seed"])
    val_dl = DataLoader(FixedSet(P["val_csv"]), batch_size=16, num_workers=2)

    while state["epoch"] < tc["max_epochs"] and state["bad"] < tc["early_stop_patience"]:
        ep = state["epoch"]
        train_ds.epoch = ep
        dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"],
                        drop_last=True)
        model.train()
        t0, agg = time.time(), []
        for i, (noisy, target, lab) in enumerate(dl):
            noisy, target, lab = noisy.to(dev), target.to(dev), lab.to(dev)
            loss, parts, _ = run_batch(model, noisy, target, lab, loss_fn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            if not (torch.isfinite(loss) and torch.isfinite(gn)):  # e.g. a GPU command buffer killed by the OS
                log.info(f"ep {ep} step {i}: non-finite loss/grad, step skipped")
                continue
            opt.step()
            agg.append({"loss": float(loss), **parts})
            if i % 50 == 0:
                log.info(f"ep {ep} step {i}/{len(dl)} loss {float(loss):.4f} {parts} {time.time() - t0:.0f}s")
        tr = pd.DataFrame(agg).mean().to_dict()

        model.eval()
        vagg = []
        with torch.no_grad():
            for noisy, target, lab in val_dl:
                noisy, target, lab = noisy.to(dev), target.to(dev), lab.to(dev)
                loss, parts, est = run_batch(model, noisy, target, lab, loss_fn)
                vagg.append({"loss": float(loss), "si_sdr": float(si_sdr(est, target).mean()), **parts})
        va = pd.DataFrame(vagg).mean().to_dict()
        sched.step(va["loss"])
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        state["history"].append(row)
        improved = va["loss"] < state["best"]
        state["best"] = min(state["best"], va["loss"])
        state["bad"] = 0 if improved else state["bad"] + 1
        state["epoch"] += 1
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "state": state, "cfg": cfg, "n_params": n_params}
        torch.save(ck, f"{out}/last.pt")
        if improved:
            torch.save({"model": model.state_dict(), "cfg": cfg, "n_params": n_params, "epoch": ep,
                        "val": va}, f"{out}/best.pt")
        pd.DataFrame(state["history"]).to_csv(f"{out}/history.csv", index=False)
        log.info(f"epoch {ep} done: train {tr['loss']:.4f} val {va['loss']:.4f} val_si_sdr {va['si_sdr']:.2f} dB "
                 f"{'*best*' if improved else ''} ({row['time_s']:.0f}s)")
    json.dump({"best_val_loss": state["best"], "epochs": state["epoch"], "params": n_params},
              open(f"{out}/summary.json", "w"), indent=1)
    return out
