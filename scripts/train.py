"""Train one experiment from config/config.yaml.

    python scripts/train.py --exp E_dacrn_film            # full run
    python scripts/train.py --exp E_dacrn_film --resume   # continue from runs/<exp>/last.pt
    python scripts/train.py --exp E_dacrn_film --demo     # tiny CPU-friendly run
    python scripts/train.py --all                          # every experiment, sequentially
    python scripts/train.py --all --reverse                # a second queue from the other end (parallel GPU use)

Queues claim experiments with runs/<exp>/.lock (holding the PID), so several queues never train the
same experiment; a lock whose process is gone is taken over and the run resumes from last.pt.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml  # noqa: E402

from src.training.trainer import load_config, train  # noqa: E402


def claim(exp):
    os.makedirs(f"runs/{exp}", exist_ok=True)
    lock = f"runs/{exp}/.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(open(lock).read() or 0), 0)
            return False  # another live queue owns it
        except (ProcessLookupError, ValueError):
            os.remove(lock)  # stale
            return claim(exp)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    if not a.all:
        train(load_config(a.exp, a.config, a.demo), resume=a.resume)
        sys.exit()
    exps = list(yaml.safe_load(open(a.config))["experiments"])
    for e in reversed(exps) if a.reverse else exps:
        if os.path.exists(f"runs/{e}/summary.json") or not claim(e):
            continue  # finished, or owned by another queue
        try:
            train(load_config(e, a.config, a.demo), resume=True)
        finally:
            os.remove(f"runs/{e}/.lock")
