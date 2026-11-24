"""Phase 4: streaming evaluation of the converted Akida model against the onset-pair baseline.

    docker run --rm -v $REPO:/repo -w /repo/clap_training -v ~/Desktop/akidaHomeKit_data:/data -e CLAP_DATA=/data clap-train python evaluate.py

Slides the 1 s window over the val and test 30 s blocks of every recording at the board's
60 ms block step (ungated, see the ledger), with the KWS sketch's decision logic: smoothed
score (alpha 0.7), STREAK consecutive scores over the threshold, then a refractory period.

- Double claps: nicla_utils' my_double_claps() pairs whose span fits in a block. A trigger
  within HIT_WINDOW_S after the second clap is a hit.
- False triggers: triggers outside every clap slot zone. Inside a zone but not a hit
  (TV pair, half pair, missed slot) they are unscored, as in the dataset builder.
- Threshold chosen on val only (highest with val recall >= TARGET_RECALL); test is reported once.
- Baseline: check_recording's pairs(onset_times()) (1.5 to 4 kHz onsets, 0.12 to 0.8 s gap),
  trigger at the second onset, same refractory and scoring.
"""

import sys
from collections import defaultdict
from pathlib import Path

import akida
import numpy as np

import data
import frontend

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nicla_utils"))
import build_dataset  # noqa: E402
import check_recording  # noqa: E402

MODEL = Path("cache/run/clap.fbz")
STEP_FRAMES = 3  # one inference per 60 ms block
ALPHA = 0.7
STREAK = 3
REFRACTORY_S = 1.5
HIT_WINDOW_S = 1.5
ZONE_S = (-0.5, 2.0)  # around every clap slot: triggers here are not false triggers
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
TARGET_RECALL = 0.9
FRAME_S = frontend.HOP / frontend.SAMPLE_RATE_HZ


def decide(ends: np.ndarray, probs: np.ndarray, threshold: float) -> list[float]:
    """Trigger times from per-block scores, as the board would fire."""
    smoothed, streak, quiet_until, out = 0.0, 0, -1.0, []
    for t, p in zip(ends, probs):
        if t < quiet_until:
            continue
        smoothed = ALPHA * p + (1 - ALPHA) * smoothed
        streak = streak + 1 if smoothed >= threshold else 0
        if streak >= STREAK:
            out.append(float(t))
            smoothed, streak, quiet_until = 0.0, 0, t + REFRACTORY_S
    return out


def refractory(times: list[float]) -> list[float]:
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= REFRACTORY_S:
            out.append(t)
    return out


def score(triggers: list[float], claps: list[tuple[float, float]], zones: list[tuple[float, float]]) -> tuple[int, list[float]]:
    """(claps hit, false trigger times)."""
    hit = {i for t in triggers for i, (s, g) in enumerate(claps) if s + g <= t <= s + g + HIT_WINDOW_S}
    false = [t for t in triggers if not any(a <= t <= b for a, b in zones)]
    return len(hit), false


def collect(model: akida.Model) -> list[dict]:
    """One entry per val/test block: scores, claps, zones and baseline triggers."""
    out = []
    recordings = sorted(p.stem for p in (data.DATA / "recordings").glob("*.wav"))
    for rec in recordings:
        feats = data.stream_features(rec)
        _, x = check_recording.load(data.DATA / "recordings" / f"{rec}.wav")
        claps = build_dataset.my_double_claps(x) if rec.startswith("doubleclaps") else []
        slots = build_dataset.clap_slots(claps) if claps else []
        zones = [(t + ZONE_S[0], t + ZONE_S[1]) for t in slots]
        baseline = [s + g for s, g in check_recording.pairs(check_recording.onset_times(x))]
        for split in ("val", "test"):
            for f0, f1 in data.blocks(feats, split):
                starts = np.arange(f0, f1 - frontend.N_FRAMES + 1, STEP_FRAMES)
                if len(starts) == 0:
                    continue
                windows = np.stack([feats[s : s + frontend.N_FRAMES] for s in starts])[..., None]
                logits = model.predict(windows.astype(np.uint8)).reshape(len(starts), -1)
                e = np.exp(logits - logits.max(1, keepdims=True))
                t0, t1 = f0 * FRAME_S, f1 * FRAME_S
                block_zones = [(max(a, t0), min(b, t1)) for a, b in zones if b > t0 and a < t1]
                out.append({
                    "recording": rec.rsplit("_", 2)[0], "split": split, "t0": t0, "t1": t1,
                    "ends": (starts * frontend.HOP + frontend.WINDOW_SAMPLES) / frontend.SAMPLE_RATE_HZ,
                    "probs": (e / e.sum(1, keepdims=True))[:, 1],
                    "claps": [(s, g) for s, g in claps if s - 0.5 >= t0 and s + g + HIT_WINDOW_S <= t1],
                    "zones": block_zones,
                    "scored_s": (t1 - t0) - sum(b - a for a, b in block_zones),
                    "baseline": refractory([t for t in baseline if t0 <= t <= t1]),
                })
    return out


def tally(blocks: list[dict], split: str, threshold: float | None) -> dict:
    """Per recording: claps, hits, false trigger times, scored hours. threshold None = baseline."""
    rows = defaultdict(lambda: {"claps": 0, "hits": 0, "false": [], "hours": 0.0})
    for b in blocks:
        if b["split"] != split:
            continue
        trig = b["baseline"] if threshold is None else decide(b["ends"], b["probs"], threshold)
        hits, false = score(trig, b["claps"], b["zones"])
        r = rows[b["recording"]]
        r["claps"] += len(b["claps"])
        r["hits"] += hits
        r["false"] += false
        r["hours"] += b["scored_s"] / 3600
    return dict(rows)


def totals(rows: dict) -> tuple[int, int, int, float]:
    claps = sum(r["claps"] for r in rows.values())
    return claps, sum(r["hits"] for r in rows.values()), sum(len(r["false"]) for r in rows.values()), sum(r["hours"] for r in rows.values())


def report(name: str, rows: dict) -> None:
    print(f"\n{name}")
    print(f"  {'recording':36s} {'claps':>5s} {'hits':>5s} {'false':>6s} {'scored min':>10s} {'false/h':>8s}")
    for rec, r in sorted(rows.items()):
        print(f"  {rec:36s} {r['claps']:5d} {r['hits']:5d} {len(r['false']):6d} {r['hours'] * 60:10.1f} {len(r['false']) / r['hours']:8.1f}")
    claps, hits, false, hours = totals(rows)
    print(f"  {'total':36s} {claps:5d} {hits:5d} {false:6d} {hours * 60:10.1f} {false / hours:8.1f}")


def main() -> None:
    blocks = collect(akida.Model(str(MODEL)))
    print("threshold sweep on val:")
    chosen = None
    for th in THRESHOLDS:
        claps, hits, false, hours = totals(tally(blocks, "val", th))
        print(f"  {th:.2f}: recall {hits}/{claps}, false {false} ({false / hours:.1f}/h)")
        if claps and hits / claps >= TARGET_RECALL:
            chosen = th
    if chosen is None:
        chosen = max(THRESHOLDS, key=lambda th: (totals(tally(blocks, "val", th))[1], -th))
        print(f"  no threshold reaches {TARGET_RECALL:.0%} recall on val")
    print(f"chosen threshold {chosen}")
    for split in ("val", "test"):
        report(f"{split}: model @ {chosen}", tally(blocks, split, chosen))
        report(f"{split}: baseline (onset pairs)", tally(blocks, split, None))
    review = data.DATA / "review"
    review.mkdir(exist_ok=True)
    print(f"\nmodel false triggers, 3 s each (trigger at 2 s) in {review}:")
    for split in ("val", "test"):
        for rec, r in sorted(tally(blocks, split, chosen).items()):
            stem = next((data.DATA / "recordings").glob(f"{rec}_*.wav")).stem
            pcm = data.load_wav(data.DATA / "recordings" / f"{stem}.wav")
            for t in r["false"]:
                name = f"false_{split}_{rec}_{check_recording.mmss(t).replace(':', 'm')}s.wav"
                first = int((t - 2) * frontend.SAMPLE_RATE_HZ)
                build_dataset.write_clip(review / name, pcm[max(0, first) : first + 3 * frontend.SAMPLE_RATE_HZ])
                print(f"  {name}")


if __name__ == "__main__":
    main()
