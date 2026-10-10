"""Live microphone -> enhancement -> speaker loop (PortAudio via sounddevice).

The model runs inside the audio callback on blocks of HOP samples, so there is no extra queueing:
latency = block buffering + algorithmic (32 ms window) + compute + device buffers. Late callbacks
(compute > block duration) are counted as xruns, so real-time behaviour is measured, not assumed.
"""
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from src.preprocessing.dsp import HOP, SR
from src.streaming.engine import StreamingEnhancer


class LiveSession:
    def __init__(self, model, input_device=None, output_device=None, gain_db=0.0, play=True, max_record_s=60):
        torch.set_num_threads(1)  # one frame at a time: threading overhead would dominate
        self.enh = StreamingEnhancer(model)
        self.gain = 10 ** (gain_db / 20)
        self.play, self.bypass = play, False
        self.devices = (input_device, output_device)
        self.max_rec = int(max_record_s * SR)
        self.rec_in, self.rec_out = [], []
        self.n_rec = 0
        self.xruns = 0
        self.status_flags = 0
        self.level_in = self.level_out = -90.0
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, outdata, frames, t, status):
        if status:
            self.status_flags += 1
        t0 = time.perf_counter()
        x = indata[:, 0] * self.gain
        self.enh.bypass = self.bypass
        y = self.enh.process(x)
        if self.play:
            outdata[:, 0] = y
        else:
            outdata.fill(0)
        if time.perf_counter() - t0 > frames / SR:
            self.xruns += 1
        if self.n_rec < self.max_rec:
            with self.lock:
                self.rec_in.append(x.copy())
                self.rec_out.append(y.copy())
                self.n_rec += frames
        self.level_in = 10 * np.log10(np.mean(x ** 2) + 1e-10)
        self.level_out = 10 * np.log10(np.mean(y ** 2) + 1e-10)

    def start(self):
        kind = (self.devices[0], self.devices[1] if self.play else None)
        if self.play:
            self.stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=1, dtype="float32",
                                    latency="low", device=kind, callback=self._callback)
        else:  # capture only (e.g. no output device): still processed live
            self.stream = sd.InputStream(samplerate=SR, blocksize=HOP, channels=1, dtype="float32",
                                         latency="low", device=self.devices[0],
                                         callback=lambda i, f, t, s: self._callback(i, np.zeros_like(i), f, t, s))
        self.stream.start()
        return self

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def recordings(self):
        with self.lock:
            cat = lambda a: np.concatenate(a) if a else np.zeros(0, np.float32)  # noqa: E731
            return cat(self.rec_in), cat(self.rec_out)

    def stats(self):
        s = self.enh.stats()
        lat = self.stream.latency if self.stream is not None else (0.0, 0.0)
        if not isinstance(lat, (tuple, list)):
            lat = (lat, 0.0)
        dev_ms = 1000 * (lat[0] + (lat[1] if self.play else 0.0))
        s.update(device_latency_ms=dev_ms, block_ms=HOP / SR * 1000, xruns=self.xruns,
                 status_flags=self.status_flags, level_in_db=self.level_in, level_out_db=self.level_out,
                 end_to_end_latency_ms=s["algorithmic_latency_ms"] + s["buffering_latency_ms"] + s["mean_ms"] + dev_ms,
                 recorded_s=self.n_rec / SR)
        return s
