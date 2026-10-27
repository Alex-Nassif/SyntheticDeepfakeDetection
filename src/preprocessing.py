import io
from pathlib import Path
from typing import Union
 
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
 
# Match these to whatever the model was trained with.
TARGET_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # CLIP
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)  # CLIP

# "crop": resize the shorter side to TARGET_SIZE, then center crop (CLIP style).
# "pad":  shrink to fit and fill the rest with black bars (letterbox).
RESIZE_MODE = "crop"
PAD_COLOR = (0, 0, 0)
 
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
            if isinstance(source, bytes):
                data = source
            elif hasattr(source, "getvalue"):
                # Streamlit uploads: getvalue() returns the whole file even if
                # something like st.image already read it.
                data = source.getvalue()
            else:
                data = source.read()
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

def to_rgb(img: Image.Image) -> Image.Image:
    """Convert grayscale, palette, CMYK, or transparent images to plain RGB."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, PAD_COLOR)
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")
 
 
def resize_to_square(img: Image.Image, mode: str = RESIZE_MODE) -> Image.Image:
    """Make the image TARGET_SIZE x TARGET_SIZE using the chosen method."""
    size = (TARGET_SIZE, TARGET_SIZE)
    if mode == "crop":
        return ImageOps.fit(img, size, method=Image.Resampling.BICUBIC,
                            centering=(0.5, 0.5))
    if mode == "pad":
        return ImageOps.pad(img, size, method=Image.Resampling.BICUBIC,
                            color=PAD_COLOR, centering=(0.5, 0.5))
    raise ValueError(f"Unknown resize mode: {mode}")

def normalize(arr: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 normalized with MEAN and STD."""
    arr = arr.astype(np.float32) / 255.0          # 0 to 255 -> 0 to 1
    arr = (arr - MEAN) / STD                      # per-channel normalization
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # channels first
 
 
def preprocess_image(source: ImageInput):
    """Prepare one image for the model.
 
    Returns:
        array: float32 array of shape (3, TARGET_SIZE, TARGET_SIZE), normalized.
        info:  dict of metadata useful for tests and debugging.
    """
    img = load_image(source)
    original_size = img.size  # (width, height)
    original_mode = img.mode
 
    img = ImageOps.exif_transpose(img)  # fix phone photo orientation
    img = to_rgb(img)
 
    if img.width == 0 or img.height == 0:
        raise ImagePreprocessingError("The image has no pixels.")
 
    img = resize_to_square(img)
    arr = normalize(np.asarray(img, dtype=np.uint8))
 
    if not np.isfinite(arr).all():
        raise ImagePreprocessingError("Preprocessing produced invalid values.")
 
    info = {
        "original_size": original_size,
        "original_mode": original_mode,
        "resize_mode": RESIZE_MODE,
        "output_shape": arr.shape,
    }
    return arr, info