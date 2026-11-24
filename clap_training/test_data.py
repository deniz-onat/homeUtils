# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""uv run test_data.py: window slicing and split blocks agree with nicla_utils/build_dataset.py."""

import numpy as np

import data

feats = np.arange(16_000 * 10, dtype=np.int64).reshape(-1, 10)  # 16,000 frames = 320 s
w = data.window_at(feats, 1.015)  # 50.75 frames -> 51 (rounded, not floored)
assert w.shape == (49, 10) and w[0, 0] == 510
assert data.window_at(feats, 319.5) is None  # runs past the end
val, test = data.blocks(feats, "val"), data.blocks(feats, "test")
assert val == [(12_000, 13_500)] and test == [(13_500, 15_000)], (val, test)  # 30 s blocks 8 and 9
assert data.blocks(feats, "train")[-1] == (15_000, 16_000)  # block 10 is train again, cut short
print("ok")

# mixed_window: with the partner silenced, the mix must equal the stream slice (alignment check)
import frontend

rng = np.random.default_rng(1)
t = np.arange(16000 * 4) / 16000
pcm = (-1750 + 400 * rng.standard_normal(t.size) + 3000 * np.sin(2 * np.pi * 300 * t) * (t % 1 < 0.1)).astype(np.int16)
stream = frontend.features_akida(frontend.dc_block(pcm))
for start_s in (0.5, 1.013, 2.0):
    want = data.window_at(stream, start_s)
    got = data.mixed_window(pcm, start_s, pcm, 1.0, 1.0, 0.0)
    assert got.shape == (49, 10), got.shape
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.max() <= 1, (start_s, diff.max())  # DC blocker state after 0.2 s warm-up vs whole stream
assert data.mixed_window(pcm, 0.1, pcm, 1.0, 1.0, 0.0) is None  # no room for the warm-up
print("mixed_window ok")
