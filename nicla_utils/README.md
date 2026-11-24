# nicla_utils

Tools for the Nicla Vision + BrainBoard1500 (akidaHomeKit).

| Path | What it is |
|---|---|
| `recorder/recorder.ino` | Streams the onboard microphone over USB on request; the OLED shows `REC` or `IDLE` (the red LED is lit while recording). |
| `record.py` | Mac side: starts and stops the recording and writes a WAV. |
| `test_record.py` | Check for the frame parser: `uv run test_record.py`. |
| `check_recording.py` | Checks WAVs: dropped audio, noise floor, clipping, transients per hour, double claps. |
| `test_check_recording.py` | Detector check on synthetic audio: `uv run test_check_recording.py`. |
| `build_dataset.py` | Cuts all recordings into 1 s clips in `dataset/<split>/<label>/` plus `manifest.csv` (rules in its docstring). |
| `test_build_dataset.py` | Labelling and split check on synthetic audio: `uv run test_build_dataset.py`. |

## Flash

```bash
FQBN=arduino:mbed_nicla:nicla_vision
PORT=/dev/cu.usbmodemXXXX   # ls /dev/cu.usbmodem*
arduino-cli core install arduino:mbed_nicla
arduino-cli lib install U8g2
arduino-cli compile --fqbn $FQBN recorder
arduino-cli upload --fqbn $FQBN --port $PORT recorder
```

## Record

```bash
uv run record.py --port $PORT --label bg_empty                  # until Ctrl-C
uv run record.py --port $PORT --label claps_door --minutes 5   # fixed length
```

WAVs go to `recordings/<label>_<YYYY-MM-DD_HH-MM-SS>.wav` (16 kHz, mono, int16, about 115 MB per hour). Each run
prints peak level, clipped samples and dropped buffers. The audio is raw: no DC
removal, so the training side must apply the same front end as the KWS example.

## Check

```bash
uv run check_recording.py recordings/<file>.wav [...]
```

## Dataset

```bash
uv run build_dataset.py   # rebuilds dataset/ from recordings/ (git-ignored, ~0.5 GB)
```

## Protocol

- Mac to board: the lines `START` and `STOP`. Closing the port also stops a recording.
- Board to Mac, only while recording: 1028-byte frames made of the magic `A5 5A`,
  a uint16 LE count of buffers dropped since `START`, then 512 int16 LE samples (32 ms).
