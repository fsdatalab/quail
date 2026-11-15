"""Small PNG and JPEG images written in a test."""

import io
import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png(width: int, height: int, shade: int = 255) -> bytes:
    """Return an 8-bit RGB PNG of one gray shade.

    Args:
        width: The width in pixels.
        height: The height in pixels.
        shade: Each pixel's red, green, and blue value.
    """
    row = b"\x00" + bytes([shade]) * (3 * width)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(row * height))
            + _chunk(b"IEND", b""))


def jpeg(width: int, height: int, shade: int = 255) -> bytes:
    """Return a JPEG of one gray shade.

    Args:
        width: The width in pixels.
        height: The height in pixels.
        shade: Each pixel's red, green, and blue value.
    """
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (width, height), (shade,) * 3).save(out, format="JPEG")
    return out.getvalue()
