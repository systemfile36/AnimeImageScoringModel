import tensorflow as tf
from tensorflow.keras import layers


def data_augmentation(x, 
        zoom_range: None | tuple[float, float], rotation_range: None | tuple[float, float],
        flip: None | str = "horizontal",
        seed: int = 42
    ):
    """
    Add Data augmentation layer to Layer 'x'.

    It's active at training only (when calls to Model.fit)
    """
    
    if zoom_range is not None:
        x = layers.RandomZoom(zoom_range, seed=seed)(x)

    if rotation_range is not None:
        x = layers.RandomRotation(rotation_range, seed=seed)(x)

    if flip is not None:
        x = layers.RandomFlip(flip, seed=seed)(x)

    return x
