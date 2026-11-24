# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Detector check on synthetic audio: uv run test_check_recording.py"""

import numpy as np

from check_recording import SAMPLE_RATE_HZ, on_rhythm, onset_times, pairs, rhythm_slots, transient_times

rng = np.random.default_rng(0)
x = rng.normal(0, 300, 60 * SAMPLE_RATE_HZ)  # 60 s of noise
burst = rng.normal(0, 8000, 400) * np.exp(-np.arange(400) / 80)  # 25 ms decaying clap
for start_s in np.arange(5.0, 56.0, 5.0):  # 11 double claps, 0.25 s apart
    for t in (start_s, start_s + 0.25):
        i = int(t * SAMPLE_RATE_HZ)
        x[i : i + 400] += burst
x[int(30.3 * SAMPLE_RATE_HZ) : int(30.3 * SAMPLE_RATE_HZ) + 400] += burst  # a third clap is not a new pair

found = on_rhythm(pairs(onset_times(x)))
assert len(found) == 11, found
assert all(abs(g - 0.25) < 0.02 for _, g in found), found
assert all(abs(s - e) < 0.02 for (s, _), e in zip(found, np.arange(5.0, 56.0, 5.0))), found
assert rhythm_slots(found) == 11
assert len(transient_times(x)) == 23, len(transient_times(x))  # 22 pair claps + the stray one
print("ok")
