"""Streaming (frame-by-frame, cached state) must reproduce offline whole-utterance inference."""
import numpy as np
import pytest
import torch

from src.models.dacrn import DACRN
from src.models.nsnet2 import NSNet2
from src.preprocessing.dsp import istft, stft
from src.streaming.engine import enhance_streaming


@pytest.mark.parametrize("make", [lambda: DACRN(cond_mode="film"), lambda: DACRN(cond_mode="none"), NSNet2])
@pytest.mark.parametrize("block", [256, 160, 1024])
def test_streaming_matches_offline(make, block):
    torch.manual_seed(0)
    model = make().eval()
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    with torch.no_grad():
        offline = model(torch.from_numpy(x)[None])[0][0].numpy()
    online, stats = enhance_streaming(model, x, block=block)
    assert online.shape == offline.shape
    np.testing.assert_allclose(online, offline, atol=1e-4)
    assert stats["frames"] > 0


def test_stft_roundtrip():
    x = torch.randn(3, 12345)
    assert torch.allclose(istft(stft(x), x.shape[-1]), x, atol=1e-5)
