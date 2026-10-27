import io
from pathlib import Path
from typing import Union
 
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
 
# Match these to whatever the model was trained with.
TARGET_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_COLOR = (0, 0, 0)  # letterbox color for non-square images
 
ImageInput = Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
 
 
class ImagePreprocessingError(Exception):
    """Raised when an input cannot be turned into a valid model input."""
 
 
def load_image(source: ImageInput) -> Image.Image:
    """Load an image from a path, raw bytes, a file-like object, or an array.
 
    Accepts the object Streamlit's file_uploader returns, since it behaves
    like a BytesIO.
    """
    if isinstance(source, Image.Image):
        return source
 
    if isinstance(source, np.ndarray):
        if source.ndim not in (2, 3) or source.size == 0:
            raise ImagePreprocessingError(f"Unsupported array shape {source.shape}")
        return Image.fromarray(source.astype(np.uint8))
 
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise ImagePreprocessingError(f"File not found: {path}")
            if path.stat().st_size == 0:
                raise ImagePreprocessingError("The image file is empty.")
            img = Image.open(path)
        else:
            data = source if isinstance(source, bytes) else source.read()
            if not data:
                raise ImagePreprocessingError("The image file is empty.")
            img = Image.open(io.BytesIO(data))
        img.load()  # force a full decode so truncated files fail here
    except ImagePreprocessingError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImagePreprocessingError(
            "The file could not be read as an image. It may be corrupted "
            "or not an image."
        ) from exc
    return img
 
 
def preprocess_image(source: ImageInput, normalize: bool = True):
    """Prepare one image for the model.
 
    Returns:
        array: float32 array of shape (3, TARGET_SIZE, TARGET_SIZE).
        info:  dict of metadata useful for tests and debugging.
    """
    img = load_image(source)
    original_size = img.size  # (width, height)
    original_mode = img.mode
 
    # Respect camera orientation tags so phone photos aren't sideways.
    img = ImageOps.exif_transpose(img)
 
    # Grayscale, palette, CMYK and RGBA all become plain 3-channel RGB.
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, PAD_COLOR)
            background.paste(img, mask=img.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")
 
    if img.width == 0 or img.height == 0:
        raise ImagePreprocessingError("The image has no pixels.")
 
    # Scale to fit inside a square and pad the rest, keeping proportions.
    img = ImageOps.pad(
        img,
        (TARGET_SIZE, TARGET_SIZE),
        method=Image.Resampling.BICUBIC,
        color=PAD_COLOR,
        centering=(0.5, 0.5),
    )
 
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3), 0 to 1
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3, H, W)
 
    if not np.isfinite(arr).all():
        raise ImagePreprocessingError("Preprocessing produced invalid values.")
 
    info = {
        "original_size": original_size,
        "original_mode": original_mode,
        "output_shape": arr.shape,
        "normalized": normalize,
    }
    return arr, info
 