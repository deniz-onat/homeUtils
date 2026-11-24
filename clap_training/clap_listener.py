# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial"]
# ///
"""Run an Apple Shortcut whenever clap_detector reports a double clap.

    uv run clap_listener.py --shortcut "Clap Lights" [--port /dev/cu.usbmodem2401] [--dry-run]

Reads clap_detector's serial lines; each DOUBLECLAP line runs `shortcuts run <name>`.
--dry-run only logs detections (for counting false triggers without switching lights).
Without --port it uses the first /dev/cu.usbmodem*. When the board is unplugged or
reflashed it waits and reconnects. Only one program can hold the port, so stop this
before board_check.py, record.py or an upload.
"""

import argparse
import glob
import subprocess
import time
from collections.abc import Callable
from datetime import datetime

RETRY_S = 5.0
SHORTCUT_TIMEOUT_S = 30.0


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def handle(line: str, on_clap: Callable[[], None]) -> None:
    if line.startswith("DOUBLECLAP"):
        log(line)
        on_clap()
    elif line.startswith("[clap]") and not line.startswith("[clap] alive"):
        log(line)  # boot and failure lines; heartbeats stay out of the log


def run_shortcut(name: str) -> None:
    try:
        result = subprocess.run(["shortcuts", "run", name], capture_output=True, text=True, timeout=SHORTCUT_TIMEOUT_S)
        if result.returncode != 0:
            log(f"shortcut {name!r} failed: {result.stderr.strip()}")
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"shortcut {name!r} failed: {e}")


def main() -> None:
    import serial

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortcut", required=True, help="name of the Apple Shortcut that turns the lights on")
    ap.add_argument("--port", help="serial port (default: first /dev/cu.usbmodem*)")
    ap.add_argument("--dry-run", action="store_true", help="log detections, run nothing")
    args = ap.parse_args()
    on_clap = (lambda: None) if args.dry_run else (lambda: run_shortcut(args.shortcut))
    log(f"listening for double claps -> {'(dry run)' if args.dry_run else repr(args.shortcut)}")

    waiting_logged = False
    while True:
        port = args.port or next(iter(sorted(glob.glob("/dev/cu.usbmodem*"))), None)
        if port is None:
            if not waiting_logged:
                log("no board found, waiting")
                waiting_logged = True
            time.sleep(RETRY_S)
            continue
        try:
            with serial.Serial(port, timeout=1.0) as s:
                log(f"connected to {port}")
                waiting_logged = False
                while True:
                    handle(s.readline().decode(errors="replace").strip(), on_clap)
        except (serial.SerialException, OSError) as e:
            log(f"{port}: {e}; retrying in {RETRY_S:.0f} s")
            time.sleep(RETRY_S)


if __name__ == "__main__":
    main()
