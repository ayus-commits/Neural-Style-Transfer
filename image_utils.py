"""
Image validation and sanitization utilities.

These functions are the first line of defense against malicious or malformed
uploads: oversized files, decompression bombs, corrupted images, unsupported
formats, and embedded alpha/orientation metadata that could break the NST
pipeline downstream. Nothing here trusts the browser/client.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("nst_app")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 4096  # px, per side
MIN_DIMENSION = 8  # px, reject degenerate images

# Hard cap on total decoded pixel count so PIL refuses to even decode a
# decompression bomb (e.g. a 50000x50000 PNG that's a few KB on disk).
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION + 1_000_000


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation. Message is safe to show to users."""


def validate_extension(filename: str) -> str:
    """Reject anything that isn't jpg/jpeg/png by filename extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '.{ext}'. Allowed types: jpg, jpeg, png."
        )
    return ext


def validate_size(data: bytes) -> None:
    """Reject empty files and anything over the 10 MB cap, before we ever decode it."""
    if not data:
        raise ImageValidationError("Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"File too large ({len(data) / 1_000_000:.1f} MB). Maximum allowed is 10 MB."
        )


def sanitize_image(data: bytes, dest_path: Path) -> Tuple[int, int]:
    """
    Validate integrity, enforce max resolution, strip alpha channels, normalize
    EXIF orientation, and write a clean RGB JPEG to dest_path.

    Raises ImageValidationError (safe to show the user) on any problem:
    corrupted file, decompression bomb, oversized dimensions, bad format, etc.
    """
    # Pass 1: verify() checks structural integrity without fully decoding pixel
    # data. PIL guarantees the file object is unusable after verify(), so we
    # must reopen a fresh handle for the real decode below.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("File is not a valid image.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image rejected: exceeds maximum allowed pixel count.") from exc
    except Exception as exc:  # noqa: BLE001 - any structural corruption funnels here
        raise ImageValidationError("Image appears to be corrupted.") from exc

    # Pass 2: real decode now that structure is trusted.
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode now, while we still control the error
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image rejected: exceeds maximum allowed pixel count.") from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageValidationError("Image could not be decoded.") from exc

    width, height = img.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        img.close()
        raise ImageValidationError(
            f"Image resolution {width}x{height} exceeds the maximum of "
            f"{MAX_DIMENSION}x{MAX_DIMENSION}."
        )
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        img.close()
        raise ImageValidationError("Image is too small to process.")

    # Normalize orientation using EXIF tags (so a photo taken sideways doesn't
    # silently get processed sideways), then drop EXIF entirely on save.
    img = ImageOps.exif_transpose(img)

    # Flatten alpha / palette-transparency onto a white background so the NST
    # pipeline always receives a clean 3-channel RGB tensor.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path, format="JPEG", quality=95)
    final_size = img.size

    # Release PIL buffers explicitly rather than waiting on GC.
    img.close()
    del img

    return final_size
