# /// script
# requires-python = ">=3.11"
# dependencies = ["pyserial"]
# ///
"""Frame-parser check: uv run test_record.py"""

import struct

from record import FRAME_BYTES, MAGIC, PAYLOAD_BYTES, split_frames


def frame(overruns: int, fill: int) -> bytes:
    return MAGIC + struct.pack("<H", overruns) + bytes([fill]) * PAYLOAD_BYTES


# Boot text before the first frame is skipped; a trailing partial frame waits.
buffer = bytearray(b"boot text\r\n" + frame(0, 1) + frame(3, 2) + frame(3, 3)[:100])
frames = split_frames(buffer)
assert [(o, p[0]) for o, p in frames] == [(0, 1), (3, 2)]
assert len(buffer) == 100

# The rest of the partial frame completes it.
buffer += frame(3, 3)[100:]
assert [(o, p[0]) for o, p in split_frames(buffer)] == [(3, 3)]
assert buffer == b""

# Audio that contains the magic bytes is not mistaken for a frame start.
tricky = MAGIC + struct.pack("<H", 0) + (MAGIC * (PAYLOAD_BYTES // 2))
buffer = bytearray(tricky + frame(1, 9))
assert [(o, p[:2]) for o, p in split_frames(buffer)] == [(0, MAGIC), (1, bytes([9, 9]))]
assert len(tricky) == FRAME_BYTES

print("ok")
