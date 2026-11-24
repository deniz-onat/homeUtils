# clap_training

Double-clap detector for the Nicla Vision + BrainBoard1500 (akidaHomeKit): BrainChip's
Akida 1.0 DS-CNN KWS fine-tuned on recordings from `nicla_utils`, run on the AKD1500,
switching the lights through an Apple Shortcut.

| Path | What it is |
|---|---|
| `frontend.py` | The board's audio front end in numpy, plus the remap to BrainChip's input (`quantize_akida`). |
| `data.py` | Model inputs from the recordings and the dataset manifest; audio-domain mixing. |
| `parity_check.py` | Phase 1 gate: pretrained model on board features vs BrainChip's pipeline. |
| `train.py` | Fine-tune, convert to Akida 1.0 (`cache/run/clap.fbz`). |
| `evaluate.py` | Streaming evaluation vs the onset-pair baseline; writes false-trigger clips. |
| `export_bb15.py` | Program, metadata and `feature_constants.h` into `firmware/clap_detector/`. |
| `firmware/clap_detector/` | The board sketch. |
| `board_check.py` | On-board check: board features vs `frontend.py` on the same audio. |
| `clap_listener.py` | Mac side: `DOUBLECLAP` on serial runs a Shortcut. |
| `test_*.py` | Checks: `uv run test_<name>.py` (host, numpy only). |

## Train (MacBook or any arm64/x86 Linux-capable Docker host)

akida 2.5.0 (the engine the BB15 library loads) has no macOS wheels, so this runs in Docker.
Data lives in `~/Desktop/akidaHomeKit_data` (`recordings/`, `dataset/manifest.csv` from
`nicla_utils/build_dataset.py`).

```bash
docker build -t clap-train .
D="-v $HOME/Desktop/akidaHomeKit_data:/data -e CLAP_DATA=/data"
docker run --rm -v $PWD:/work $D clap-train python train.py
docker run --rm -v $PWD/..:/repo -w /repo/clap_training $D clap-train python evaluate.py
KWS=$HOME/REPOS/Personal/akida-face-unlock/third_party/brainboard1500_arduino_library/examples/bb15_nicla_vision_keyword_spotting
docker run --rm -v $PWD:/work -v $KWS:/kws:ro -e KWS_EXAMPLE=/kws clap-train python export_bb15.py
```

`parity_check.py` and `kws_reference.py` need `cache/speech_commands` (Speech Commands
v0.02 test set), `cache/models/ds_cnn_kws_iq8_wq4_aq4_laq1.h5` and `cache/kws_preprocessed.pkl`
(BrainChip); `train.py` needs the model file. All are downloads, see the docstrings.

## Flash (Mac mini, board on USB)

```bash
FQBN=arduino:mbed_nicla:nicla_vision
PORT=/dev/cu.usbmodem2401   # arduino-cli board list
LIB=$HOME/REPOS/Personal/akida-face-unlock/third_party/brainboard1500_arduino_library
arduino-cli compile --fqbn $FQBN --library $LIB firmware/clap_detector
arduino-cli upload --fqbn $FQBN --port $PORT firmware/clap_detector
```

The model is compiled into the sketch (21 KB); nothing is written to BB15 flash.
Boot log: `[clap] akida_ready ...` then `[clap] listening`. A red blink means setup failed;
the serial log says which stage. LED green for 1.5 s after each detection.

Serial commands (any terminal, or `arduino-cli monitor -p $PORT`): `SCORES` toggles a
score line per 60 ms inference, `DUMP` is what `board_check.py` uses.

## Check on the board

```bash
uv run board_check.py --port $PORT    # clap or talk during the 3 s; PASS = same bytes as training
```

## Listener

Make a Shortcut that turns the lights on (name it e.g. `Clap Lights`), then:

```bash
uv run clap_listener.py --shortcut "Clap Lights" --dry-run   # count detections only
uv run clap_listener.py --shortcut "Clap Lights"
```

At login, via launchd:

```bash
mkdir -p ~/Library/LaunchAgents
cat > ~/Library/LaunchAgents/local.clap-listener.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.clap-listener</string>
  <key>ProgramArguments</key><array>
    <string>$(which uv)</string><string>run</string><string>clap_listener.py</string>
    <string>--shortcut</string><string>Clap Lights</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/clap-listener.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/clap-listener.log</string>
</dict></plist>
EOF
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.clap-listener.plist
# stop (before flashing or board_check): launchctl bootout gui/$(id -u)/local.clap-listener
```
