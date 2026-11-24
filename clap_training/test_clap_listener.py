# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial"]
# ///
"""uv run test_clap_listener.py: only DOUBLECLAP lines run the shortcut."""

import clap_listener

calls = []
lines = [
    "[clap] alive ms=30000 inferences=500 detections=0 overruns=0",
    "S 0.120 0.090",
    "DOUBLECLAP score=0.973 ms=123456",
    "noise DOUBLECLAP",
    "[clap] result=FAIL stage=akida",
    "DOUBLECLAP score=0.981 ms=130000",
]
for line in lines:
    clap_listener.handle(line, lambda: calls.append(1))
assert len(calls) == 2, calls
print("ok")
