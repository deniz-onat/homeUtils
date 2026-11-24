# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Cut the recordings into 1 s clips for fine-tuning the KWS model on double claps.

    uv run build_dataset.py        # recordings/*.wav -> dataset/ (rebuilt from scratch)

Layout (Speech Commands style): dataset/<split>/<label>/<recording>_<start ms>.wav and
dataset/manifest.csv. Labels: double_clap, negative. Clips are raw (no DC removal), as
the board delivers them; the training front end must match the board's.

- Positives: my double claps in doubleclaps_* recordings. Train gets several shifted
  windows per clap (the streaming window can catch the pair anywhere); val and test get
  one centred window, so their counts are real claps.
- Negatives: every other 1 s window (0.5 s hop) of every recording. In clap sessions,
  the time around each expected clap slot is skipped, so a missed or half-detected clap
  is never labelled negative. Half pairs are therefore dropped, not used as negatives.
- Splits: contiguous 30 s blocks (8 train, 1 val, 1 test of every 10), never random
  windows; windows crossing a block edge are dropped, so overlap cannot leak.
  The test split here is provisional; the real test is a held-out day.
"""

import csv
import shutil
import wave
from pathlib import Path

import numpy as np

from check_recording import SAMPLE_RATE_HZ, load, on_rhythm, onset_times, pairs, transient_times

RECORDINGS = Path("recordings")
DATASET = Path("dataset")
WINDOW_S = 1.0  # model input: 49 MFCC frames at a 20 ms hop
HOP_S = 0.5
BLOCK_S = 30.0
SPLIT_OF_BLOCK = ["train"] * 8 + ["val", "test"]
GAP_TOLERANCE_S = 0.08  # my gap is 0.22 +- 0.04 s; TV pairs rarely land this close to my median
PAIR_LEAD_S = 0.05  # room before the first clap
PAIR_TAIL_S = 0.15  # second clap's decay kept inside the window
SLOT_ZONE_S = (-0.5, 1.0)  # around every expected clap slot: no negatives
NOMINAL_PERIOD_S = 5.0
TRAIN_SHIFTS = 5


def my_double_claps(x: np.ndarray) -> list[tuple[float, float]]:
    """(first clap s, gap s) per ~5 s slot.

    ponytail: TV pairs are rejected only by rhythm and by a gap close to my median; a TV
    pair that matches both becomes a false positive. Listen-and-label if the model shows it.
    """
    found = on_rhythm(pairs(onset_times(x)))
    if not found:
        return []
    median_gap = np.median([g for _, g in found])
    kept: list[tuple[float, float]] = []
    for start, gap in found:
        if abs(gap - median_gap) > GAP_TOLERANCE_S:
            continue
        if kept and start - kept[-1][0] < NOMINAL_PERIOD_S / 2:
            if abs(gap - median_gap) < abs(kept[-1][1] - median_gap):
                kept[-1] = (start, gap)
        else:
            kept.append((start, gap))
    return kept


def clap_slots(claps: list[tuple[float, float]]) -> list[float]:
    """Detected clap times plus the slots between them where a clap was expected but not found."""
    starts = [s for s, _ in claps]
    steps = [b - a for a, b in zip(starts, starts[1:]) if 0.8 * NOMINAL_PERIOD_S <= b - a <= 1.2 * NOMINAL_PERIOD_S]
    period = float(np.median(steps)) if steps else NOMINAL_PERIOD_S
    slots = []
    for a, b in zip(starts, starts[1:]):
        missing = max(1, round((b - a) / period))
        slots += [a + j * (b - a) / missing for j in range(missing)]
    return slots + starts[-1:]


def split_of(seconds: float) -> str:
    return SPLIT_OF_BLOCK[int(seconds // BLOCK_S) % len(SPLIT_OF_BLOCK)]


def plan_windows(duration_s: float, claps: list[tuple[float, float]], slots: list[float]) -> list[tuple[float, str, str]]:
    """(start s, label, split) for every clip to cut from one recording."""
    windows = []
    for start, gap in claps:
        earliest = start + gap + PAIR_TAIL_S - WINDOW_S
        latest = start - PAIR_LEAD_S
        split = split_of(start)
        starts = np.linspace(earliest, latest, TRAIN_SHIFTS) if split == "train" else [(earliest + latest) / 2]
        windows += [(s, "double_clap", split) for s in starts if s >= 0 and s + WINDOW_S <= duration_s]

    zones = [(t + SLOT_ZONE_S[0], t + SLOT_ZONE_S[1]) for t in slots]
    for s in np.arange(0.0, duration_s - WINDOW_S + 1e-9, HOP_S):
        crosses_block = int(s // BLOCK_S) != int((s + WINDOW_S - 1e-9) // BLOCK_S)
        near_clap = any(s < b and s + WINDOW_S > a for a, b in zones)
        if not crosses_block and not near_clap:
            windows.append((float(s), "negative", split_of(s)))
    return windows


def write_clip(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(samples.astype("<i2").tobytes())


def main() -> None:
    shutil.rmtree(DATASET, ignore_errors=True)
    rows = []
    for recording in sorted(RECORDINGS.glob("*.wav")):
        raw, x = load(recording)
        duration_s = len(raw) / SAMPLE_RATE_HZ
        claps = my_double_claps(x) if recording.stem.startswith("doubleclaps") else []
        slots = clap_slots(claps) if claps else []
        transients = transient_times(x)
        for start, label, split in plan_windows(duration_s, claps, slots):
            first = int(round(start * SAMPLE_RATE_HZ))
            clip = DATASET / split / label / f"{recording.stem}_{int(round(start * 1000)):07d}.wav"
            write_clip(clip, raw[first : first + SAMPLE_RATE_HZ])
            has_transient = bool(((transients >= start) & (transients < start + WINDOW_S)).any())
            rows.append([clip.relative_to(DATASET), split, label, recording.stem, f"{start:.3f}", int(has_transient)])
        print(f"{recording.stem}: {len(claps)} double claps, {sum(r[3] == recording.stem for r in rows)} clips")

    with open(DATASET / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([["path", "split", "label", "recording", "start_s", "has_transient"], *rows])
    for split in ("train", "val", "test"):
        counts = {label: sum(r[1] == split and r[2] == label for r in rows) for label in ("double_clap", "negative")}
        print(f"{split:5s}  double_clap {counts['double_clap']:5d}  negative {counts['negative']:6d}")


if __name__ == "__main__":
    main()
