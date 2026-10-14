import numpy as np
import torch
from PIL import Image, ImageFilter

def get_input(input_array: np.ndarray):
    """
    Defining input file and loading.
    Expected to receive multidimensional array as:
    - Channels
    - Height
    - Width
    """


    shape = tuple(input_array.shape)
    if len(shape) != 3:
        raise ValueError(
            f"Expected format of (channels, height, width). Received: {shape}"
        )

    
    channels, height, width = shape

    return {
        "content": input_array,
        "metadata": {
            "shape": shape,
            "channels": channels,
            "height": height,
            "width": width,
            #returns either the dtype itself or unknown
            "dtype": str(getattr(input_array, "dtype", "unknown")), 
        },
    }





def processing_input(
    input_data,
    model,
    pixel_range="0_255",
    high_pass=True,
):
    """
    Prepare a CHW image array for GenD inference.

    Args:
        input_data: Dictionary returned by get_input().
        model: Loaded GenD model.
        pixel_range: Expected input pixel range ("0_255" or "0_1").
        high_pass: Whether to apply a high-pass filter.

    Returns:
        Dictionary containing the model input tensor and metadata.
    """

    # Get the image and convert it to a NumPy array.
    image_array = np.asarray(input_data["content"])

    if image_array.ndim != 3:
        raise ValueError("Expected a 3D image array (C, H, W).")

    # Convert from channel-first (C, H, W) to image format (H, W, C).
    if image_array.shape[0] not in (1, 3, 4):
        raise ValueError(
            "Expected 1, 3, or 4 channels in the first dimension."
        )

    image_array = np.transpose(image_array, (1, 2, 0))

    # Convert grayscale or RGBA images to RGB.
    if image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Convert pixel values to uint8.
    if pixel_range == "0_255":
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    elif pixel_range == "0_1":
        image_array = np.clip(image_array, 0, 1) * 255
        image_array = image_array.astype(np.uint8)
    else:
        raise ValueError("pixel_range must be '0_255' or '0_1'.")

    image = Image.fromarray(image_array, mode="RGB")

    # Apply a high-pass filter by subtracting a blurred version
    # of the image from the original, then shifting the result
    # to the middle of the 8-bit intensity range.
    if high_pass:
        original = np.asarray(image).astype(np.float32)
        blurred = np.asarray(
            image.filter(ImageFilter.GaussianBlur(radius=2))
        ).astype(np.float32)

        high_pass_array = original - blurred + 128
        high_pass_array = np.clip(
            high_pass_array, 0, 255
        ).astype(np.uint8)

        image = Image.fromarray(high_pass_array, mode="RGB")

    # Use GenD's own preprocessing for resizing and normalization.
    tensor = model.feature_extractor.preprocess(image)

    # Add batch dimension: (C, H, W) -> (1, C, H, W).
    tensor = tensor.unsqueeze(0)

    # Preserve the original input metadata.
    metadata = dict(input_data["metadata"])
    metadata["high_pass_applied"] = high_pass
    metadata["model_input_shape"] = tuple(tensor.shape)

    return {
        "data": tensor,
        "metadata": metadata,
    }



