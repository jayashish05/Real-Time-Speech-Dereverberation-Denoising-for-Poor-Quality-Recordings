"""Real-time microphone enhancement from the terminal (use headphones to avoid feedback).

    python scripts/realtime.py                      # mic -> DA-CRN -> speakers, stats every second
    python scripts/realtime.py --list-devices
    python scripts/realtime.py --no-play --seconds 10 --save results/live   # capture + save wavs
    python scripts/realtime.py --benchmark           # streaming latency/RTF of every model, no audio device
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from src.inference.enhancer import DEFAULT_MODEL, available_models, load_model  # noqa: E402
from src.models.dacrn import denormalise  # noqa: E402


def benchmark():
    import torch
    torch.set_num_threads(1)
    from src.streaming.engine import enhance_streaming
    x = np.random.default_rng(0).standard_normal(16000 * 10).astype(np.float32) * 0.05
    print(f"{'model':28s} {'params':>9s} {'mean ms/frame':>14s} {'p95':>7s} {'RTF':>6s}")
    for name in available_models():
        m = load_model(name)
        if m.cond_mode == "oracle":
            continue
        _, s = enhance_streaming(m, x)
        print(f"{name:28s} {m.info['params']:9,d} {s['mean_ms']:14.2f} {s['p95_ms']:7.2f} {s['rtf']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input", type=int)
    ap.add_argument("--output", type=int)
    ap.add_argument("--gain-db", type=float, default=0.0)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl-C")
    ap.add_argument("--save", help="folder for input/enhanced wavs")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    if a.benchmark:
        return benchmark()
    import sounddevice as sd
    if a.list_devices:
        return print(sd.query_devices())
    from src.streaming.live import LiveSession
    sess = LiveSession(load_model(a.model), a.input, a.output, a.gain_db, play=not a.no_play,
                       max_record_s=max(a.seconds, 60)).start()
    print("streaming... Ctrl-C to stop")
    t0 = time.time()
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            time.sleep(1)
            s = sess.stats()
            est = sess.enh.estimate
            e = "" if est is None else " | est SNR %.0f dB T60 %.2f s C50 %.0f dB" % tuple(
                float(v) for v in denormalise(np.asarray(est)).values())
            print(f"compute {s['mean_ms']:.2f} ms/frame (p95 {s['p95_ms']:.2f}, RTF {s['rtf']:.3f}) "
                  f"latency {s['end_to_end_latency_ms']:.0f} ms xruns {s['xruns']} in {s['level_in_db']:.0f} dB{e}")
            if s["level_in_db"] < -90:
                print("  ! microphone delivers digital silence: check the OS microphone permission for this terminal")
    except KeyboardInterrupt:
        pass
    sess.stop()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        x, y = sess.recordings()
        sf.write(f"{a.save}/live_input.wav", x, 16000)
        sf.write(f"{a.save}/live_enhanced.wav", y, 16000)
        print("saved to", a.save)


if __name__ == "__main__":
    main()
