import tensorflow as tf 
from tensorflow.keras import layers, Model

from src.utils import get_filename_based_logger

logger = get_filename_based_logger(__file__)

def create_resnet152_pretrained(input: layers.Input, trainable: bool=False):
    """
    Create pre-trained ResNet152 from keras.applications and return output.

    `include_top` is False
    """
   
    # Check whether shape is match to ResNet
    if input.shape[1:] != (224, 224, 3): # compare without batch
        logger.error(f"Invalid Input shape : {input.shape}. It should be (224, 224, 3) for pretrained ResNet")
        return None

    # Include preprocess layer for ResNet. 
    # See https://keras.io/api/applications/resnet/#resnet152-function
    x = layers.Lambda(
        lambda x: tf.keras.applications.resnet.preprocess_input(tf.cast(x, tf.float32))
    )

    base_model = tf.keras.applications.ResNet152(
        include_top=False,
        pooling='avg',
        input_tensor=input, 
        weights='imagenet'
    )
    base_model.trainable = trainable

    return base_model.output

def create_efficientnet_b7_pretrained(input: layers.Input, trainable: bool=False):
    """
    Create per-trained EfficientNetB7 from keras.applications and return output

    `input.shape` should match (600, 600, 3) 

    `include_top` is False
    """

    if input.shape[1:] != (600, 600, 3):
        logger.error(f"Invalid Input shape : {input.shape}. It should be (600, 600, 3) for pre-trained EfficientNetB7")

    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_tensor=input,
        pooling='avg'
    )
    base_model.trainable = trainable

    return base_model.output

