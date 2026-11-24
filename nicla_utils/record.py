# /// script
# requires-python = ">=3.11"
# dependencies = ["pyserial"]
# ///
"""Record raw 16 kHz audio from the Nicla Vision recorder sketch into a WAV file.

Sends START, writes audio until Ctrl-C or --minutes, then sends STOP.
Reports peak level, clipped samples and dropped buffers.
"""

import argparse
import math
import struct
import time
import wave
from array import array
from datetime import datetime
from pathlib import Path

import serial

SAMPLE_RATE_HZ = 16000
MAGIC = b"\xa5\x5a"
HEADER_BYTES = 4  # magic + uint16 LE overruns since START
PAYLOAD_BYTES = 1024  # kPdmBufferBytes in recorder.ino
FRAME_BYTES = HEADER_BYTES + PAYLOAD_BYTES
SAMPLES_PER_FRAME = PAYLOAD_BYTES // 2
FULL_SCALE = 32768
REPORT_INTERVAL_S = 5.0


def split_frames(buffer: bytearray) -> list[tuple[int, bytes]]:
    """Remove complete frames from the front of buffer; return (overruns, payload) pairs.

    Bytes before the first magic (boot text, a partial frame) are dropped. Frames
    then arrive back to back, so the magic sits at offset 0 and audio that happens
    to contain the magic bytes is never searched.
    """
    frames = []
    while True:
        start = buffer.find(MAGIC)
        if start < 0:
            del buffer[: max(len(buffer) - 1, 0)]  # last byte may be half a magic
            return frames
        del buffer[:start]
        if len(buffer) < FRAME_BYTES:
            return frames
        (overruns,) = struct.unpack_from("<H", buffer, 2)
        frames.append((overruns, bytes(buffer[HEADER_BYTES:FRAME_BYTES])))
        del buffer[:FRAME_BYTES]


def dbfs(peak: int) -> float:
    return 20 * math.log10(peak / FULL_SCALE) if peak else -math.inf


def record(port: serial.Serial, wav: wave.Wave_write, deadline: float) -> None:
    buffer = bytearray()
    frames = peak = clipped = overruns = 0
    next_report = time.monotonic() + REPORT_INTERVAL_S
    try:
        while time.monotonic() < deadline:
            buffer += port.read(port.in_waiting or 1)
            for overruns, payload in split_frames(buffer):
                samples = array("h", payload)  # little-endian on both the Nicla and the Mac
                peak = max(peak, max(samples), -min(samples))
                clipped += sum(1 for s in samples if s in (FULL_SCALE - 1, -FULL_SCALE))
                wav.writeframes(payload)
                frames += 1
            if time.monotonic() >= next_report:
                next_report += REPORT_INTERVAL_S
                seconds = frames * SAMPLES_PER_FRAME / SAMPLE_RATE_HZ
                print(f"\r{seconds:8.0f} s  peak {dbfs(peak):6.1f} dBFS  clipped {clipped}  dropped {overruns}", end="")
    except KeyboardInterrupt:
        pass  # Ctrl-C is the normal way to stop
    port.write(b"STOP\n")

    seconds = frames * SAMPLES_PER_FRAME / SAMPLE_RATE_HZ
    print(f"\n{seconds:.1f} s recorded, peak {dbfs(peak):.1f} dBFS, {clipped} clipped samples")
    if clipped:
        print("Clipping: lower kPdmGain in recorder.ino or move further from the board.")
    if overruns:
        # Dropped buffers are missing from the WAV, not zero-filled, so later audio shifts earlier.
        print(f"WARNING: {overruns} buffers dropped on the board ({overruns * 32} ms of audio missing).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="e.g. /dev/cu.usbmodem1101")
    parser.add_argument("--minutes", type=float, help="stop after this long (default: until Ctrl-C)")
    parser.add_argument("--out", type=Path, default=Path("recordings"), help="output folder")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.wav"
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else math.inf

    with serial.Serial(args.port, timeout=0.1) as port, wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        port.reset_input_buffer()
        port.write(b"START\n")
        print(f"Recording to {path}; Ctrl-C to stop.")
        record(port, wav, deadline)


if __name__ == "__main__":
    main()
