"""Decode research CT inputs and preserve the uploaded slice's pixel grid."""
import gzip
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

MAX_BYTES = 128 * 1024**2
MAX_SIDE = 2048


def read_volume(data, compressed):
    import nibabel as nib

    if compressed:
        with gzip.GzipFile(fileobj=BytesIO(data)) as stream:
            data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("The uncompressed NIfTI exceeds 128 MiB. Export a smaller volume or slice.")
    cls = nib.Nifti2Image if int.from_bytes(data[:4], "little") == 540 or int.from_bytes(data[:4], "big") == 540 else nib.Nifti1Image
    volume = cls.from_bytes(data)
    if len(volume.shape) not in (2, 3) or any(n < 1 or n > 2048 for n in volume.shape):
        raise ValueError("Upload a 2D or 3D NIfTI CT with valid dimensions.")
    return volume


def window(slice_hu):
    raw = np.asarray(slice_hu, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("The selected CT slice contains NaN or infinite values.")
    return (np.clip(raw, -175, 250) + 175) / 425


def read_raster(data):
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG"} or max(image.size) > 2048:
            raise ValueError("Use a PNG or JPEG slice of up to 2048 pixels per side.")
        if image.mode not in {"1", "L", "LA", "RGB", "RGBA", "P"}:
            raise ValueError("Export this image as an 8-bit windowed PNG or JPEG, or upload the NIfTI CT.")
        return np.asarray(image.convert("L"), dtype=np.float32) / 255


def preview(gray, point, mask=None):
    pixels = np.rint(gray * 255).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=-1)
    if mask is not None:
        rgb[mask] = (0.48 * rgb[mask] + 0.52 * np.array([32, 224, 160])).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.thumbnail((800, 800))
    x, y = point
    x = int((x + 0.5) * image.width / pixels.shape[1])
    y = int((y + 0.5) * image.height / pixels.shape[0])
    draw = ImageDraw.Draw(image)
    for line in ((x - 8, y, x + 8, y), (x, y - 8, x, y + 8)):
        draw.line(line, fill="#fbbf24", width=2)
    return image


def png_mask(mask):
    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()
