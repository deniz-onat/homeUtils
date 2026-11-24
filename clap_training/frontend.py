"""The board's audio front end in numpy, so training sees the bytes the board feeds Akida.

Mirrors bb15_nicla_vision_keyword_spotting: remove_dc_offset() and quantize_feature() in
the .ino, mfcc.cpp for the coefficients. Checked against that C code by test_frontend.py.
"""

import numpy as np

SAMPLE_RATE_HZ = 16000
FRAME = 640  # 40 ms
HOP = 320  # 20 ms
N_FRAMES = 49  # model input is 49 x 10 x 1, about 1 s
N_COEFFS = 10
N_MELS = 40
MEL_LOW_HZ, MEL_HIGH_HZ = 20, 4000
MFCC_FULL_SCALE = np.float32(123.56967163085938)
DC_ALPHA_Q15 = 32700
WINDOW_SAMPLES = HOP * (N_FRAMES - 1) + FRAME  # 16000


def dc_block(pcm: np.ndarray) -> np.ndarray:
    """The board's integer DC blocker, state starting at zero like after START.

    Exact integer emulation: its floor shift leaves a DC bias of a few hundred LSB that
    reaches the lowest mel band, so a float high-pass would not match.
    ponytail: pure Python loop, about 0.2 s per 100 s of audio; fine for hours of recordings.
    """
    out = np.empty(len(pcm), np.int16)
    prev_in = prev_out = 0
    for i, x in enumerate(pcm.tolist()):
        y = x - prev_in + ((DC_ALPHA_Q15 * prev_out) >> 15)
        prev_in, prev_out = x, y
        out[i] = 32767 if y > 32767 else -32768 if y < -32768 else y
    return out


def _mel(f):
    return np.float32(1127.0) * np.log(np.float32(1.0) + np.float32(f) / np.float32(700.0), dtype=np.float32)


def _filter_bank() -> np.ndarray:
    """Dense (N_MELS, FRAME // 2 + 1) version of mfcc.cpp's packed triangular filters."""
    fb = np.zeros((N_MELS, FRAME // 2 + 1), np.float32)
    lo, hi = _mel(MEL_LOW_HZ), _mel(MEL_HIGH_HZ)
    delta = (hi - lo) / np.float32(N_MELS + 1)
    bin_mel = _mel(np.float32(SAMPLE_RATE_HZ / FRAME) * np.arange(FRAME // 2, dtype=np.float32))  # Nyquist never scanned
    for b in range(N_MELS):
        left, centre, right = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
        inside = (bin_mel > left) & (bin_mel < right)
        m = bin_mel[inside]
        fb[b, : FRAME // 2][inside] = np.where(m <= centre, (m - left) / (centre - left), (right - m) / (right - centre))
    return fb


_WINDOW = (0.5 - 0.5 * np.cos(np.float32(2 * np.pi) * np.arange(FRAME, dtype=np.float32) / np.float32(FRAME))).astype(np.float32)
_FB = _filter_bank()
_DCT = (np.sqrt(2.0 / N_MELS, dtype=np.float32) * np.cos(np.pi / N_MELS * (np.arange(N_MELS) + 0.5)[None, :] * np.arange(N_COEFFS)[:, None])).astype(np.float32)


def mfcc(pcm: np.ndarray) -> np.ndarray:
    """Float coefficients (n_frames, 10) for DC-blocked int16 PCM; frames at a 320 hop."""
    n = (len(pcm) - FRAME) // HOP + 1
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = (pcm[idx].astype(np.float32) / np.float32(32768)) * _WINDOW
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    log_mel = np.log(power @ _FB.T + np.float32(1e-6), dtype=np.float32)
    return log_mel @ _DCT.T


def quantize(coeffs: np.ndarray) -> np.ndarray:
    """quantize_feature(): float coefficient to the model's uint8 input, truncating."""
    return np.clip((coeffs / MFCC_FULL_SCALE + 1) * 128, 0, 255).astype(np.uint8)


def features(pcm: np.ndarray) -> np.ndarray:
    """uint8 features (n_frames, 10) exactly as the KWS sketch feeds its model."""
    return quantize(mfcc(pcm))


# BrainChip's DS-CNN KWS was trained on TF's MFCC stretched to uint8 over this float range
# (element 7 of their preprocessed pickle). The board's MFCC maps onto TF's per coefficient
# with this affine fit (parity_check.py --fit; TF uses magnitude, the board power).
AKIDA_MFCC_MIN, AKIDA_MFCC_MAX = -39.44708440780639, 16.3944679546356
AKIDA_SCALE = np.array([0.639468, 0.518333, 0.503039, 0.492063, 0.486501, 0.494212, 0.468435, 0.47841, 0.455667, 0.434374], np.float32)
AKIDA_OFFSET = np.array([12.090559, -2.119687, -0.431076, -0.591429, -0.427775, -0.486408, -0.420991, -0.45877, -0.424744, -0.428157], np.float32)


def quantize_akida(coeffs: np.ndarray) -> np.ndarray:
    """Board MFCC to the uint8 input BrainChip's pretrained model expects; the clap firmware must do the same."""
    tf_like = coeffs * AKIDA_SCALE + AKIDA_OFFSET
    return np.clip(np.round((tf_like - AKIDA_MFCC_MIN) / (AKIDA_MFCC_MAX - AKIDA_MFCC_MIN) * 255), 0, 255).astype(np.uint8)


def features_akida(pcm: np.ndarray) -> np.ndarray:
    """uint8 model features (n_frames, 10) for training and the clap firmware."""
    return quantize_akida(mfcc(pcm))
