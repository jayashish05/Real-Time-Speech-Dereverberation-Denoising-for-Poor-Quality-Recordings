"""Frame-by-frame streaming enhancement.

Microphone -> blocks of any size -> FIFO -> every HOP (16 ms) new samples: window last N_FFT samples ->
rFFT -> model.stream_step (cached conv frames + GRU states) -> iFFT -> overlap-add -> HOP output samples.

The output stream lags the input by HOP samples, plus HOP samples of block buffering: an input sample
leaves the system at most N_FFT samples (32 ms) later, before compute and audio-device latency. The
output is sample-identical to the offline model output shifted by HOP (tests/test_streaming.py).
"""
import time
from collections import deque

import numpy as np
import torch

from src.preprocessing.dsp import HOP, N_FFT, window


class StreamingEnhancer:
    def __init__(self, model, bypass=False):
        self.model = model.eval().cpu()
        self.win = window().numpy()
        self.bypass = bypass
        self.reset()

    def reset(self):
        self.state = {}
        self.frame = np.zeros(N_FFT, np.float32)  # last N_FFT input samples
        self.tail = np.zeros(N_FFT - HOP, np.float32)  # overlap-add memory
        self.fifo_in, self.fifo_out = np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.times = deque(maxlen=2000)  # per-frame compute time (s)
        self.estimate = None  # latest normalised degradation estimate [3]
        self.extra_delay = 0  # zeros inserted at start-up when blocks are not a multiple of HOP

    def _frame(self, hop_samples):
        t0 = time.perf_counter()
        self.frame = np.concatenate([self.frame[HOP:], hop_samples])
        X = np.fft.rfft(self.frame * self.win)
        if self.bypass:
            Y = X
        else:
            Xt = torch.from_numpy(X.astype(np.complex64))[None, :, None]
            Yt, est, self.state = self.model.stream_step(Xt, self.state)
            Y = Yt[0, :, 0].numpy()
            if est is not None:
                self.estimate = est[0, -1].numpy()
        y = np.fft.irfft(Y, N_FFT).astype(np.float32) * self.win
        out = self.tail[:HOP] + y[:HOP]
        self.tail = np.concatenate([self.tail[HOP:], np.zeros(HOP, np.float32)]) + y[HOP:]
        self.times.append(time.perf_counter() - t0)
        return out

    def process(self, block):
        """Feed any number of samples; returns as many output samples as input (zeros while priming)."""
        self.fifo_in = np.concatenate([self.fifo_in, np.asarray(block, np.float32).ravel()])
        outs = []
        while len(self.fifo_in) >= HOP:
            outs.append(self._frame(self.fifo_in[:HOP]))
            self.fifo_in = self.fifo_in[HOP:]
        if outs:
            self.fifo_out = np.concatenate([self.fifo_out, *outs])
        n = len(block)
        if len(self.fifo_out) < n:  # only at start-up when the block is not a multiple of HOP
            self.extra_delay += n - len(self.fifo_out)
            self.fifo_out = np.concatenate([np.zeros(n - len(self.fifo_out), np.float32), self.fifo_out])
        out, self.fifo_out = self.fifo_out[:n], self.fifo_out[n:]
        return out

    def stats(self):
        t = np.array(self.times) * 1000 if self.times else np.zeros(1)
        frame_ms = HOP / 16
        return {"frames": len(self.times), "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)),
                "max_ms": float(t.max()), "rtf": float(t.mean() / frame_ms),
                "algorithmic_latency_ms": N_FFT / 16, "buffering_latency_ms": self.extra_delay / 16}


def enhance_streaming(model, x, block=HOP):
    """Run a whole signal through the streaming path (used for tests and latency benchmarks)."""
    se = StreamingEnhancer(model)
    n = len(x)
    pad = N_FFT + block + (-(n + N_FFT + block)) % block  # flush, and keep every block full-sized
    x = np.concatenate([np.asarray(x, np.float32), np.zeros(pad, np.float32)])
    y = np.concatenate([se.process(x[i:i + block]) for i in range(0, len(x), block)])
    d = HOP + se.extra_delay
    return y[d: d + n], se.stats()
