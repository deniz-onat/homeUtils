"""Model inputs from the recordings and nicla_utils' dataset manifest.

Each recording goes through the board front end once, as one continuous stream from
START (frontend.dc_block state carries over, as on the board). Training clips and
streaming windows are then 49-frame slices of that stream, so both see what an ungated
board would. Clip starts are rounded to the 20 ms frame grid (at most 10 ms shift).
"""

import csv
import wave
from pathlib import Path

import numpy as np

import frontend

DATA = Path.home() / "Desktop/akidaHomeKit_data"
LABELS = ("negative", "double_clap")  # class index = position


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def stream_features(recording: str, data: Path = DATA) -> np.ndarray:
    """uint8 (n_frames, 10) for a whole recording, cached in <data>/features/."""
    cache = data / "features" / f"{recording}.npy"
    if not cache.exists():
        pcm = load_wav(data / "recordings" / f"{recording}.wav")
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, frontend.features(frontend.dc_block(pcm)))
    return np.load(cache)


def window_at(feats: np.ndarray, start_s: float) -> np.ndarray | None:
    i = round(start_s * frontend.SAMPLE_RATE_HZ / frontend.HOP)
    w = feats[i : i + frontend.N_FRAMES]
    return w if len(w) == frontend.N_FRAMES else None


def clips(data: Path = DATA) -> dict[str, np.ndarray]:
    """Every manifest row as a model input: x (N, 49, 10, 1) uint8, y, split, has_transient, recording, start_s."""
    with open(data / "dataset/manifest.csv") as f:
        rows = list(csv.DictReader(f))
    streams = {r: stream_features(r, data) for r in sorted({row["recording"] for row in rows})}
    out = {k: [] for k in ("x", "y", "split", "has_transient", "recording", "start_s")}
    for row in rows:
        w = window_at(streams[row["recording"]], float(row["start_s"]))
        if w is None:  # last window of a recording may run past the final whole frame
            continue
        out["x"].append(w[..., None])
        out["y"].append(LABELS.index(row["label"]))
        out["split"].append(row["split"])
        out["has_transient"].append(int(row["has_transient"]))
        out["recording"].append(row["recording"])
        out["start_s"].append(float(row["start_s"]))
    return {k: np.array(v) for k, v in out.items()}


def blocks(feats: np.ndarray, split: str) -> list[tuple[int, int]]:
    """Frame ranges of the 30 s blocks nicla_utils/build_dataset.py gave to `split` (8 train, 1 val, 1 test of 10)."""
    order = ["train"] * 8 + ["val", "test"]
    per_block = int(30 * frontend.SAMPLE_RATE_HZ / frontend.HOP)  # 1500 frames
    return [(b * per_block, min((b + 1) * per_block, len(feats))) for b in range(-(-len(feats) // per_block)) if order[b % 10] == split]
