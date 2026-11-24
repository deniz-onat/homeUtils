# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyserial"]
# ///
"""On-board front-end check: do the board's features match frontend.py on the same audio?

    uv run board_check.py --port /dev/cu.usbmodem2401 [--save dump.npz]

Needs clap_detector flashed and nothing else holding the port (stop clap_listener first).
Sends DUMP; the board restarts its front end and streams 3 s of raw PCM ("P" lines) with
the quantized features it computed from it ("F" lines). This rebuilds the features from
the PCM with frontend.py and compares byte for byte. Clap or talk during the 3 s so the
features cover more than silence.
"""

import argparse
import sys
import time

import numpy as np

import frontend


def parse(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(raw int16 PCM, board features (n_frames, 10) uint8) from P/F lines."""
    pcm = [np.frombuffer(bytes.fromhex(line[2:]), "<i2") for line in lines if line.startswith("P ")]
    feats = [np.frombuffer(bytes.fromhex(line[2:]), np.uint8) for line in lines if line.startswith("F ")]
    return np.concatenate(pcm), np.concatenate(feats).reshape(-1, frontend.N_COEFFS)


def rebuild(pcm: np.ndarray) -> np.ndarray:
    """Features the board should have computed: DC blocker from zero, one zero hop in front."""
    stream = np.concatenate([np.zeros(frontend.HOP, np.int16), frontend.dc_block(pcm)])
    return frontend.features_akida(stream)


def dump(port_name: str, timeout_s: float = 10.0) -> list[str]:
    import serial

    with serial.Serial(port_name, timeout=0.2) as port:
        port.reset_input_buffer()
        port.write(b"DUMP\n")
        lines, deadline, started = [], time.monotonic() + timeout_s, False
        while time.monotonic() < deadline:
            line = port.readline().decode(errors="replace").strip()
            started |= line.startswith("DUMP START")
            if started and line[:2] in ("P ", "F "):
                lines.append(line)
            if line == "DUMP END":
                return lines
    sys.exit(f"no DUMP END within {timeout_s} s (is clap_detector running, and the port free?)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--save", help="write pcm and board features to this .npz")
    args = ap.parse_args()
    pcm, board = parse(dump(args.port))
    rebuilt = rebuild(pcm)[: len(board)]
    diff = np.abs(rebuilt.astype(int) - board.astype(int))
    print(f"{len(pcm) / frontend.SAMPLE_RATE_HZ:.2f} s, {len(board)} frames, peak {np.abs(pcm.astype(int)).max()}")
    print(f"exact {np.mean(diff == 0):.2%}, within 1: {np.mean(diff <= 1):.2%}, max diff {diff.max()}")
    if args.save:
        np.savez(args.save, pcm=pcm, board=board, rebuilt=rebuilt)
    ok = diff.max() <= 1 and np.mean(diff == 0) > 0.99
    print("PASS" if ok else "FAIL: board front end differs from training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
