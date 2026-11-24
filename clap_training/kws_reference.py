"""BrainChip's KWS preprocessing (akida_models.kws.preprocessing), for the Phase 1 parity work.

TF audio_spectrogram (40 ms window, 20 ms stride, magnitude squared) into audio_ops.mfcc
(defaults: 40 mel bins, 20 Hz to 4 kHz, 10 coefficients), then stretched to uint8 over the
float range stored in BrainChip's preprocessed pickle (its element 7).
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops

import frontend
import safe_pickle

PKL = Path("cache/kws_preprocessed.pkl")
MFCC_MIN, MFCC_MAX = frontend.AKIDA_MFCC_MIN, frontend.AKIDA_MFCC_MAX  # pickle element 7


def mfcc_tf(path: Path) -> np.ndarray:
    """Float (49, 10) TF MFCC of one Speech Commands wav, zero-padded to 1 s."""
    audio, rate = tf.audio.decode_wav(tf.io.read_file(str(path)), desired_channels=1, desired_samples=16000)
    spec = audio_ops.audio_spectrogram(audio, window_size=640, stride=320, magnitude_squared=True)
    return audio_ops.mfcc(spec, rate, dct_coefficient_count=10).numpy().reshape(49, 10)


def to_uint8(mfcc: np.ndarray) -> np.ndarray:
    return np.clip(np.round((mfcc - MFCC_MIN) / (MFCC_MAX - MFCC_MIN) * 255), 0, 255).astype(np.uint8)


def stored(paths: list[Path]) -> tuple[np.ndarray, list[int]]:
    """BrainChip's own uint8 features for `paths` (word/file keys found in the pickle), and their labels."""
    d = safe_pickle.load(PKL)
    index = {"/".join(p.split("/")[-2:]): i for i, p in enumerate(d[4])}
    rows = [index["/".join(Path(p).parts[-2:])] for p in paths]
    return d[0][rows, ..., 0], [int(d[1][r]) for r in rows]


def label_map() -> dict[str, int]:
    return safe_pickle.load(PKL)[6]
