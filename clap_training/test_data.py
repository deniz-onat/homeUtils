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
