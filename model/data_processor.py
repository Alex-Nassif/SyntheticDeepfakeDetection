import numpy as np

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


def processing_input():
    """
    Prepare and modify if necessary data for model.
    """


