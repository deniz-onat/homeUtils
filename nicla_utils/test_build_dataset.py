# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Dataset-planning check on synthetic audio: uv run test_build_dataset.py"""

import numpy as np

from build_dataset import BLOCK_S, TRAIN_SHIFTS, WINDOW_S, clap_slots, my_double_claps, plan_windows, split_of
from check_recording import SAMPLE_RATE_HZ

rng = np.random.default_rng(0)
duration_s = 90.0
x = rng.normal(0, 300, int(duration_s * SAMPLE_RATE_HZ))
burst = rng.normal(0, 8000, 400) * np.exp(-np.arange(400) / 80)


def add(t: float) -> None:
    i = int(t * SAMPLE_RATE_HZ)
    x[i : i + 400] += burst


pair_times = [t for t in np.arange(5.0, 86.0, 5.0) if t != 45.0]  # 16 double claps
for t in pair_times:
    add(t)
    add(t + 0.22)
add(45.0)  # a missed clap: only a single burst in the 45 s slot
add(62.0)
add(62.6)  # an off-rhythm pair with a different gap, like a TV pair

claps = my_double_claps(x)
assert [round(s) for s, _ in claps] == [round(t) for t in pair_times], claps
slots = clap_slots(claps)
assert any(abs(s - 45.0) < 0.3 for s in slots), slots  # the missed slot is filled in

windows = plan_windows(duration_s, claps, slots)
for start, label, _ in windows:
    end = start + WINDOW_S
    if label == "double_clap":
        assert any(start < t and t + 0.22 + 0.1 < end for t in pair_times), start  # whole pair inside
    else:
        assert not any(start < t + 0.5 and end > t - 0.1 for t in [*pair_times, 45.0]), start  # no clap
        assert int(start // BLOCK_S) == int((end - 1e-9) // BLOCK_S), start  # no block crossing
positives = [w for w in windows if w[1] == "double_clap"]
assert len(positives) == len(pair_times) * TRAIN_SHIFTS  # 90 s = blocks 0-2, all train
assert [split_of(s) for s in (0, 239, 240, 269, 270, 299, 300)] == ["train", "train", "val", "val", "test", "test", "train"]
print("ok")
