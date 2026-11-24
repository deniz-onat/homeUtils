"""Phases 2 and 3: fine-tune BrainChip's Akida 1.0 DS-CNN KWS into a double-clap detector.

    docker run --rm -v $PWD:/work -v ~/Desktop/akidaHomeKit_data:/data -e CLAP_DATA=/data clap-train python train.py

The pretrained checkpoint is already quantized (8-bit input, 4-bit weights and activations,
1-bit last activation), so it is fine-tuned as it is: quantization-aware from the start,
no float stage. Head: dense_5 replaced by a 2-class QuantizedDense (negative, double_clap).
Stage 1 trains the head only, stage 2 also separable_3 and separable_4 at a lower rate.

Writes cache/run/: clap.h5 (Keras), clap.fbz (Akida 1.0), probs.npz (val/test clip scores).
"""

from pathlib import Path

import akida
import cnn2snn
import numpy as np
import tensorflow as tf
from cnn2snn import AkidaVersion, load_quantized_model, set_akida_version

import data

PRETRAINED = Path("cache/models/ds_cnn_kws_iq8_wq4_aq4_laq1.h5")
OUT = Path("cache/run")
SEED = 0
MIXES_PER_POSITIVE = 4
POS_GAIN_DB = (-12.0, 3.0)  # farther claps, a little louder
NOISE_GAIN_DB = (-6.0, 6.0)
HARD_NEGATIVE_WEIGHT = 3.0  # negatives holding a sharp transient (manifest has_transient)
STAGES = [  # (trainable layers, learning rate, epochs)
    (["dense_5"], 1e-3, 8),
    (["dense_5", "separable_3", "separable_4"], 1e-4, 8),
]


def augmented_positives(c: dict, rng: np.random.Generator) -> np.ndarray:
    """Each train positive mixed into random train background windows at random gains."""
    recordings = {r: data.load_wav(data.DATA / "recordings" / f"{r}.wav") for r in np.unique(c["recording"])}
    pos = np.flatnonzero((c["split"] == "train") & (c["y"] == 1))
    neg = np.flatnonzero((c["split"] == "train") & (c["y"] == 0))
    out = []
    for i in pos:
        for _ in range(MIXES_PER_POSITIVE):
            j = rng.choice(neg)
            w = data.mixed_window(
                recordings[c["recording"][i]], c["start_s"][i], recordings[c["recording"][j]], c["start_s"][j],
                10 ** (rng.uniform(*POS_GAIN_DB) / 20), 10 ** (rng.uniform(*NOISE_GAIN_DB) / 20),
            )
            if w is not None:
                out.append(w[..., None])
    return np.array(out)


def build_model() -> tf.keras.Model:
    base = load_quantized_model(PRETRAINED)
    x = cnn2snn.QuantizedDense(2, quantizer=cnn2snn.MaxPerAxisQuantizer(bitwidth=4), name="dense_5")(base.get_layer("flatten").output)
    return tf.keras.Model(base.input, tf.keras.layers.Activation("softmax", name="act_softmax")(x))


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf.keras.utils.set_random_seed(SEED)
    c = data.clips()
    train = c["split"] == "train"
    aug = augmented_positives(c, rng)
    x = np.concatenate([c["x"][train], aug]).astype("float32")
    y = np.concatenate([c["y"][train], np.ones(len(aug), int)])
    hard = np.concatenate([c["has_transient"][train], np.zeros(len(aug), int)])
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    w = np.where(y == 1, n_neg / n_pos, np.where(hard == 1, HARD_NEGATIVE_WEIGHT, 1.0))
    print(f"train: {n_pos} positives ({len(aug)} mixed), {n_neg} negatives ({(hard == 1).sum()} hard)")

    val = c["split"] == "val"
    xv, yv = c["x"][val].astype("float32"), c["y"][val]
    wv = np.where(yv == 1, (yv == 0).sum() / max(1, (yv == 1).sum()), 1.0)

    model = build_model()
    for trainable, lr, epochs in STAGES:
        for layer in model.layers:
            layer.trainable = layer.name in trainable
        model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", weighted_metrics=["accuracy"])
        model.fit(x, y, sample_weight=w, validation_data=(xv, yv, wv), batch_size=128, epochs=epochs, shuffle=True, verbose=2)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "clap.h5", include_optimizer=False)
    with set_akida_version(AkidaVersion.v1):
        ak = cnn2snn.convert(model)
    ak.save(str(OUT / "clap.fbz"))

    probs = {}
    for split in ("val", "test"):
        m = c["split"] == split
        xs = c["x"][m]
        keras_p = model.predict(xs.astype("float32"), verbose=0)[:, 1]
        akida_p = tf.nn.softmax(ak.predict(xs.astype("uint8")).reshape(len(xs), -1)).numpy()[:, 1]
        probs[f"{split}_keras"], probs[f"{split}_akida"], probs[f"{split}_y"] = keras_p, akida_p, c["y"][m]
        agree = np.mean((keras_p > 0.5) == (akida_p > 0.5))
        print(f"{split}: keras vs akida decisions agree {agree:.2%}, max |p diff| {np.abs(keras_p - akida_p).max():.3f}")
    np.savez(OUT / "probs.npz", **probs)
    print("akida", akida.__version__, "ip", ak.ip_version, "layers", [layer.name for layer in ak.layers])


if __name__ == "__main__":
    main()
