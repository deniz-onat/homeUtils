# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Golden check: frontend.py against the board's mfcc.cpp built natively.

    uv run test_frontend.py

KWS_EXAMPLE points at the BB15 library's bb15_nicla_vision_keyword_spotting folder.
Not bit-exact by design: numpy's FFT and libm differ from kiss_fft and the M7 in the last
float bits, so a coefficient near a quantization step can land one byte apart.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import frontend

KWS = Path(os.environ.get("KWS_EXAMPLE", Path.home() / "REPOS/Personal/akida-face-unlock/third_party/brainboard1500_arduino_library/examples/bb15_nicla_vision_keyword_spotting"))
HERE = Path(__file__).parent


def reference(pcm: np.ndarray) -> np.ndarray:
    exe = Path(tempfile.gettempdir()) / "mfcc_ref"
    src = [HERE / "golden/mfcc_ref.cpp", KWS / "mfcc.cpp", KWS / "kiss_fft.c", KWS / "kiss_fftr.c"]
    subprocess.run(["c++", "-O2", "-x", "c++", "-std=c++17", "-I", str(KWS), *map(str, src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], input=pcm.astype("<i2").tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, frontend.N_COEFFS)


def test_signal() -> np.ndarray:
    """3 s: DC offset like the raw mic, room noise, a tone, two clap-like bursts, one clipped."""
    rng = np.random.default_rng(0)
    t = np.arange(3 * 16000) / 16000
    x = -1750 + 300 * rng.standard_normal(t.size) + 2000 * np.sin(2 * np.pi * 440 * t)
    for at, amp in [(1.0, 12000), (1.22, 40000)]:
        i = int(at * 16000)
        x[i : i + 800] += amp * rng.standard_normal(800) * np.exp(-np.arange(800) / 150)
    return np.clip(x, -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    pcm = test_signal()
    ref = reference(pcm)
    got = frontend.features(frontend.dc_block(pcm))
    assert got.shape == ref.shape, (got.shape, ref.shape)
    diff = np.abs(got.astype(int) - ref.astype(int))
    print(f"frames {ref.shape[0]}, exact {np.mean(diff == 0):.4%}, max diff {diff.max()}")
    assert diff.max() <= 1 and np.mean(diff == 0) > 0.999  # alpha off by one gives 0.991
    # a 1 s window is exactly the model's 49 frames
    assert frontend.features(frontend.dc_block(pcm[:16000])).shape == (49, 10)
    print("ok")
