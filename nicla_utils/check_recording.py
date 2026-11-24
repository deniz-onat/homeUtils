# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Sanity-check recorder WAVs: dropped audio, noise floor, clipping, transients, double claps.

    uv run check_recording.py recordings/doubleclaps_desk_tv_2026-11-24_21-32-53.wav [...]

Two detectors, on purpose:
- Transient rates: broadband energy jumps (2 ms frames, x5 over the previous 50 ms),
  the measure the vault's figures use. Speech barely triggers it.
- Clap finding: sudden rises of 1.5-4 kHz energy, inside the model's 20 Hz-4 kHz band
  and above most TV commentary, where claps stand out. It also fires on consonants, so
  its raw rates are not transient rates.
"""

import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE_HZ = 16000
FULL_SCALE = 32767
FRAME_SAMPLES = 256  # 16 ms analysis window
HOP_SAMPLES = 80  # 5 ms
FRAMES_PER_S = SAMPLE_RATE_HZ // HOP_SAMPLES
ONSET_BAND_HZ = (1500, 4000)
BASELINE_FRAMES = 40  # median of the previous 200 ms
ONSET_DB = 12.0
ONSET_MERGE_S = 1 / 12
PAIR_GAP_S = (0.12, 0.8)  # a double clap: both claps within the model's ~1 s window
RHYTHM_S = ((4.0, 6.0), (9.0, 11.0))  # sessions clap every ~5 s; allow one missed clap
CHUNK_FRAMES = 20000
TRANSIENT_FRAME_SAMPLES = 32  # 2 ms
TRANSIENT_BASELINE_FRAMES = 25  # 50 ms
TRANSIENT_RATIO = 5.0
TRANSIENT_MERGE_S = 0.1


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(path)) as wav:
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return raw, raw.astype(np.float64) - raw.mean()


def wall_seconds(path: Path) -> float:
    """Seconds between the start time in the file name and the last write."""
    started = datetime.strptime("_".join(path.stem.rsplit("_", 2)[-2:]), "%Y-%m-%d_%H-%M-%S")
    return path.stat().st_mtime - started.timestamp()


def band_energy(x: np.ndarray) -> np.ndarray:
    window = np.hanning(FRAME_SAMPLES)
    freqs = np.fft.rfftfreq(FRAME_SAMPLES, 1 / SAMPLE_RATE_HZ)
    in_band = (freqs >= ONSET_BAND_HZ[0]) & (freqs < ONSET_BAND_HZ[1])
    frames = sliding_window_view(x, FRAME_SAMPLES)[::HOP_SAMPLES]
    out = [  # chunked: a 90-minute file would need gigabytes in one go
        (np.abs(np.fft.rfft(frames[i : i + CHUNK_FRAMES] * window, axis=1)) ** 2)[:, in_band].sum(1)
        for i in range(0, len(frames), CHUNK_FRAMES)
    ]
    return np.concatenate(out)


def onset_times(x: np.ndarray) -> np.ndarray:
    energy = band_energy(x)
    padded = np.concatenate([np.full(BASELINE_FRAMES, energy[0]), energy])
    baseline = np.concatenate(
        [
            np.median(sliding_window_view(padded[i : i + CHUNK_FRAMES + BASELINE_FRAMES], BASELINE_FRAMES)[:-1], axis=1)
            for i in range(0, len(energy), CHUNK_FRAMES)
        ]
    )[: len(energy)]
    rise_db = 10 * np.log10((energy + 1) / (baseline + 1))
    frames = np.flatnonzero(rise_db > ONSET_DB)
    if len(frames) == 0:
        return np.array([])
    first_of_each = np.concatenate([[True], np.diff(frames) > ONSET_MERGE_S * FRAMES_PER_S])
    return frames[first_of_each] / FRAMES_PER_S


def transient_times(x: np.ndarray) -> np.ndarray:
    n = TRANSIENT_FRAME_SAMPLES
    energy = np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1))
    running = np.convolve(energy, np.ones(TRANSIENT_BASELINE_FRAMES) / TRANSIENT_BASELINE_FRAMES)
    previous = np.concatenate([np.ones(TRANSIENT_BASELINE_FRAMES), running[TRANSIENT_BASELINE_FRAMES - 1 : len(energy) - 1]])
    frames = np.flatnonzero(energy / (previous + 1) > TRANSIENT_RATIO)
    if len(frames) == 0:
        return np.array([])
    frames_per_s = SAMPLE_RATE_HZ / n
    first_of_each = np.concatenate([[True], np.diff(frames) > TRANSIENT_MERGE_S * frames_per_s])
    return frames[first_of_each] / frames_per_s


def rhythm_slots(found: list[tuple[float, float]]) -> int:
    """Distinct ~5 s clap slots holding a pair, so a TV pair beside a real clap is not counted twice."""
    starts = np.array([s for s, _ in found])
    steps = np.diff(starts)
    steps = steps[(steps >= RHYTHM_S[0][0]) & (steps <= RHYTHM_S[0][1])]
    if len(steps) == 0:
        return len(starts)
    return len(np.unique(np.round((starts - starts[0]) / np.median(steps))))


def pairs(onsets: np.ndarray) -> list[tuple[float, float]]:
    """(start s, gap s) for consecutive onsets close enough to be a double clap."""
    found, i = [], 0
    while i < len(onsets) - 1:
        gap = onsets[i + 1] - onsets[i]
        if PAIR_GAP_S[0] <= gap <= PAIR_GAP_S[1]:
            found.append((onsets[i], gap))
            i += 2
        else:
            i += 1
    return found


def on_rhythm(found: list[tuple[float, float]]) -> list[tuple[float, float]]:
    starts = np.array([s for s, _ in found])
    return [
        (s, g)
        for s, g in found
        if any(lo <= d <= hi for d in np.abs(starts - s) for lo, hi in RHYTHM_S)
    ]


def clip_events(raw: np.ndarray) -> list[float]:
    seconds = np.flatnonzero(np.abs(raw.astype(np.int32)) >= FULL_SCALE) / SAMPLE_RATE_HZ
    return [s for i, s in enumerate(seconds) if i == 0 or s - seconds[i - 1] > 1.0]


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def report(path: Path) -> None:
    raw, x = load(path)
    seconds = len(x) / SAMPLE_RATE_HZ
    hours = seconds / 3600
    rms = np.sqrt((x[: len(x) // 1600 * 1600].reshape(-1, 1600) ** 2).mean(1))  # 100 ms
    transients = transient_times(x)
    transient_pairs = pairs(transients)
    clips = clip_events(raw)

    print(f"\n{path.name}")
    print(f"  length      {seconds:.1f} s audio vs {wall_seconds(path):.1f} s wall time (a gap means dropped audio)")
    print(f"  floor       100 ms RMS median {np.median(rms):.0f}, p95 {np.percentile(rms, 95):.0f}, DC {raw.mean():.0f}")
    print(f"  clipping    {(np.abs(raw.astype(np.int32)) >= FULL_SCALE).sum()} samples in {len(clips)} events"
          + (f" at {', '.join(mmss(s) for s in clips[:10])}" if clips else ""))
    print(f"  transients  {len(transients) / hours:.0f}/h, {len(transient_pairs) / hours:.0f} double-clap-like pairs/h")
    if path.stem.startswith("doubleclaps"):
        rhythm = on_rhythm(pairs(onset_times(x)))
        gaps = sorted(g for _, g in rhythm) or [0.0]
        print(f"  claps       about {rhythm_slots(rhythm)} double claps (~5 s slots holding a pair); "
              f"gap median {gaps[len(gaps) // 2]:.2f} s, range {gaps[0]:.2f}-{gaps[-1]:.2f} s")
        print(f"              {len(rhythm)} candidate pairs, TV pairs included, at {' '.join(f'{s:.1f}' for s, _ in rhythm)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        report(Path(arg))
