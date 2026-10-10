"""Download the dataset subset described in DATASET.md (~3 GB instead of ~9 GB).

* LibriSpeech dev-clean, test-clean: full archives.
* LibriSpeech train-clean-100: only the first --train-gb GB of the gzip stream (complete speaker
  folders; a truncated last file is deleted). Archive members are ordered by speaker, so this yields
  a speaker subset, not a random utterance subset.
* SLR28: only simulated RIRs, real RIRs and point-source noises are fetched, via HTTP range requests
  on the zip (isotropic noises, ~0.5 GB, are skipped).
* DEMAND: 5 environments (16 kHz), the VoiceBank+DEMAND test environments -> unseen-noise test.

Re-runnable: finished parts are skipped. Uses only the standard library.
"""
import argparse, io, os, zlib, shutil, subprocess, tarfile, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

LS = "https://openslr.elda.org/resources/12/{}.tar.gz"
LS_PREFIX = "https://www.openslr.org/resources/12/{}.tar.gz"  # second mirror: spreads the load
SLR28 = "https://openslr.trmal.net/resources/28/rirs_noises.zip"
DEMAND = "https://zenodo.org/records/1227121/files/{}_16k.zip?download=1"
DEMAND_ENVS = ["DLIVING", "OOFFICE", "PCAFETER", "SPSQUARE", "TBUS"]
SLR28_KEEP = ("RIRS_NOISES/simulated_rirs/", "RIRS_NOISES/pointsource_noises/")


def curl(url, out, byte_range=None):
    cmd = ["curl", "-sSL", "--fail", "--retry", "50", "--retry-all-errors",
           "--speed-limit", "1000", "--speed-time", "30", "-o", out, url]  # abort+retry stalled transfers
    if byte_range:
        cmd += ["-r", byte_range]
    else:
        cmd += ["-C", "-"]
    subprocess.run(cmd, check=True)


def librispeech(raw, name, max_bytes=None):
    done = os.path.join(raw, "LibriSpeech", f".{name}.done")
    if os.path.exists(done):
        return
    if max_bytes is None:
        tgz = os.path.join(raw, f"{name}.tar.gz")
        curl(LS.format(name), tgz)
        with tarfile.open(tgz) as t:
            t.extractall(raw, filter="data")
        os.remove(tgz)
    else:  # fetch a prefix of the archive (resumable chunks) and extract what is complete
        part = os.path.join(raw, f"{name}.prefix.tar.gz")
        if not os.path.exists(part):
            open(part, "wb").close()
        _fill(part, LS_PREFIX.format(name), 0, max_bytes - 1)
        m = None
        try:
            with tarfile.open(part, mode="r|gz") as t:
                for m in t:
                    t.extract(m, raw, filter="data")
        except (EOFError, tarfile.ReadError, OSError, zlib.error):  # expected: truncated archive
            if m is not None and os.path.isfile(os.path.join(raw, m.name)):
                os.remove(os.path.join(raw, m.name))  # the member cut by the truncation
        os.remove(part)
        os.remove(part + ".chunks")
    open(done, "w").close()
    print("LibriSpeech", name, "ready")


def slr28(raw):
    """Fetch only the needed members of the remote zip: download their contiguous byte span into a
    sparse local file that also holds the central directory, then extract with zipfile."""
    if os.path.exists(os.path.join(raw, "RIRS_NOISES", ".done")):
        return
    size = int(urllib.request.urlopen(urllib.request.Request(SLR28, method="HEAD")).headers["Content-Length"])
    sparse = os.path.join(raw, "rirs_noises.partial.zip")
    tail = 16 << 20  # central directory (~7 MB for 62k members) lives at the end
    with open(sparse, "ab") as f:
        f.truncate(size)
    _fill(sparse, SLR28, size - tail, size - 1)
    with zipfile.ZipFile(sparse) as z:
        infos = z.infolist()
    offsets = sorted(i.header_offset for i in infos)
    nxt = {o: (offsets[k + 1] if k + 1 < len(offsets) else size - tail) for k, o in enumerate(offsets)}
    want = [i for i in infos if i.filename.startswith(SLR28_KEEP)
            or (i.filename.startswith("RIRS_NOISES/real_rirs_isotropic_noises/") and "noise" not in os.path.basename(i.filename))
            or i.filename.endswith(("README", "LICENSE", "ANNOTATIONS", "rir_list", "noise_list"))]
    spans = _merge([(i.header_offset, nxt[i.header_offset] - 1) for i in want])
    print(f"SLR28: fetching {sum(b - a + 1 for a, b in spans) / 1e6:.0f} MB in {len(spans)} ranges")
    for a, b in spans:
        _fill(sparse, SLR28, a, b)
    with zipfile.ZipFile(sparse) as z:
        for i in want:
            z.extract(i, raw)
    os.remove(sparse)
    os.remove(sparse + ".chunks")
    open(os.path.join(raw, "RIRS_NOISES", ".done"), "w").close()
    print("SLR28 ready")


def _merge(spans, gap=1 << 20):
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _fill(path, url, a, b, chunk=32 << 20, parallel=4):
    """Write bytes a..b of url into path at the same offsets, in parallel resumable chunks
    (servers here throttle per connection, so parallel ranges are faster)."""
    import threading
    log = path + ".chunks"
    done = [tuple(map(int, r.split("-"))) for r in open(log).read().split()] if os.path.exists(log) else []
    covered = lambda s, e: any(x <= s and e <= y for x, y in done)  # noqa: E731
    todo = [(s, min(s + chunk - 1, b)) for s in range(a, b + 1, chunk) if not covered(s, min(s + chunk - 1, b))]
    lock = threading.Lock()

    def one(se):
        s, e = se
        part = f"{path}.{s}.part"
        curl(url, part, f"{s}-{e}")
        assert os.path.getsize(part) == e - s + 1, "server ignored the Range header"
        with lock:
            with open(path, "r+b") as f, open(part, "rb") as p:
                f.seek(s)
                shutil.copyfileobj(p, f)
            os.remove(part)
            with open(log, "a") as f:
                f.write(f"{s}-{e}\n")

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, todo))


def demand(raw, env):
    dst = os.path.join(raw, "DEMAND", env)
    if os.path.isdir(dst):
        return
    z = os.path.join(raw, f"{env}.zip")
    curl(DEMAND.format(env), z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(raw, "DEMAND"))
    os.remove(z)
    print("DEMAND", env, "ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--train-gb", type=float, default=1.0, help="prefix of train-clean-100 to fetch")
    a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    jobs = [lambda: librispeech(a.raw, "dev-clean"), lambda: librispeech(a.raw, "test-clean"),
            lambda: librispeech(a.raw, "train-clean-100", int(a.train_gb * 1e9)),
            lambda: slr28(a.raw)] + [lambda e=e: demand(a.raw, e) for e in DEMAND_ENVS]
    with ThreadPoolExecutor(len(jobs)) as ex:  # parallel connections add up
        for f in [ex.submit(j) for j in jobs]:
            f.result()
    print("download complete")


if __name__ == "__main__":
    main()
