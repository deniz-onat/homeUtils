# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyserial"]
# ///
"""uv run test_board_check.py: DUMP parsing and the rebuild match a block-by-block simulation of clap_detector.ino."""

import numpy as np

import board_check
import frontend


def firmware_blocks(raw: np.ndarray) -> list[np.ndarray]:
    """clap_detector.ino after DUMP: zeroed g_mfcc_input, DC blocker from zero, 960-sample blocks."""
    mfcc_input = np.zeros(320 + 960, np.int16)
    filtered = frontend.dc_block(raw)
    out = []
    for b in range(len(raw) // 960):
        mfcc_input[320:] = filtered[b * 960 : (b + 1) * 960]
        out.append(frontend.features_akida(mfcc_input))  # 3 frames of 640 at hop 320
        mfcc_input[:320] = mfcc_input[960:]
    return out


rng = np.random.default_rng(2)
raw = (-1700 + 600 * rng.standard_normal(960 * 6)).astype(np.int16)
raw[2000:2300] += 20000
lines = []
for b, feats in enumerate(firmware_blocks(raw)):
    lines.append("P " + raw[b * 960 : (b + 1) * 960].astype("<i2").tobytes().hex())
    lines.append("F " + feats.tobytes().hex())
lines.append("DUMP END")

pcm, board = board_check.parse(lines)
assert (pcm == raw).all() and board.shape == (18, 10)
rebuilt = board_check.rebuild(pcm)
assert (rebuilt == board).all(), np.abs(rebuilt.astype(int) - board).max()
print("ok")
