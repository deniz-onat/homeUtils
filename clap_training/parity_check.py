"""Phase 1 gate: do the board's features work with BrainChip's pretrained DS-CNN KWS?

    docker run --rm --network none -v $PWD:/work:ro clap-train python parity_check.py [--fit]

Speech Commands test clips (cache/speech_commands, 10 core words) into BrainChip's Akida 1.0
DS-CNN KWS (cache/models, from data.brainchip.com/models/AkidaV1/ds_cnn/), labelled with the
map in BrainChip's preprocessed pickle. Three feature paths:

- TF pipeline: BrainChip's own preprocessing (kws_reference.py); the harness check.
- board quantizer: frontend.features(), the KWS sketch's bytes as they are.
- board + remap: frontend.features_akida(), board MFCC with the per-coefficient affine map
  to TF's MFCC (TF uses magnitude, the board power: slopes near 0.5).

--fit refits frontend.AKIDA_SCALE / AKIDA_OFFSET on even clips and prints them; the
accuracies are always on odd clips. Most of these clips are in BrainChip's training list,
so absolute numbers are optimistic; compare the paths with each other.
"""

import glob
import sys
from pathlib import Path

import numpy as np
from cnn2snn import load_quantized_model

import data
import frontend
import kws_reference

WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
PRETRAINED = Path("cache/models/ds_cnn_kws_iq8_wq4_aq4_laq1.h5")
PER_WORD = 400


def board_mfcc(path: Path) -> np.ndarray:
    pcm = data.load_wav(path)
    pcm = np.pad(pcm, (0, max(0, frontend.WINDOW_SAMPLES - len(pcm))))[: frontend.WINDOW_SAMPLES]
    return frontend.mfcc(frontend.dc_block(pcm))


def fit_remap(board: np.ndarray, tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares tf = scale * board + offset, per coefficient."""
    scale, offset = np.zeros(frontend.N_COEFFS), np.zeros(frontend.N_COEFFS)
    for i in range(frontend.N_COEFFS):
        b = board[..., i].ravel()
        (scale[i], offset[i]), *_ = np.linalg.lstsq(np.stack([b, np.ones_like(b)], 1), tf[..., i].ravel(), rcond=None)
    return scale, offset


def main() -> None:
    model = load_quantized_model(PRETRAINED)
    label_of = kws_reference.label_map()
    paths, labels = [], []
    for word in WORDS:
        files = sorted(glob.glob(f"cache/speech_commands/{word}/*.wav"))[:PER_WORD]
        paths += files
        labels += [label_of[word]] * len(files)
    labels = np.array(labels)
    tf = np.stack([kws_reference.mfcc_tf(p) for p in paths])
    board = np.stack([board_mfcc(Path(p)) for p in paths])
    fit, test = np.arange(len(paths)) % 2 == 0, np.arange(len(paths)) % 2 == 1

    if "--fit" in sys.argv:
        scale, offset = fit_remap(board[fit], tf[fit])
        print("AKIDA_SCALE =", np.array2string(scale.astype(np.float32), precision=6, separator=", "))
        print("AKIDA_OFFSET =", np.array2string(offset.astype(np.float32), precision=6, separator=", "))

    def accuracy(x: np.ndarray) -> float:
        pred = model.predict(x[test][..., None].astype("float32"), verbose=0).argmax(-1).ravel()
        return float(np.mean(pred == labels[test]))

    results = {
        "TF pipeline": accuracy(kws_reference.to_uint8(tf)),
        "board quantizer": accuracy(frontend.quantize(board)),
        "board + remap": accuracy(frontend.quantize_akida(board)),
    }
    for name, acc in results.items():
        print(f"{name:16s} {acc:.1%}  ({test.sum()} clips)")
    gap = results["TF pipeline"] - results["board + remap"]
    print(f"gate: board + remap within {gap:.1%} of TF pipeline -> {'PASS' if gap <= 0.03 else 'FAIL'} (limit 3 points)")


if __name__ == "__main__":
    main()
