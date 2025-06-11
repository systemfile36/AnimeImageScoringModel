import tensorflow as tf 
from tensorflow.keras import layers, Model, Input

from src.utils import get_filename_based_logger

from .layers import resnet_bottleneck_model

logger = get_filename_based_logger(__file__)

def create_resnet152_pretrained(input: Input, trainable: bool=False, pooling: bool=True):
    """
    Create pre-trained ResNet152 from keras.applications and return output layer.

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
    )(input)

    base_model = tf.keras.applications.ResNet152(
        include_top=False,
        pooling='avg' if pooling else None,
        input_tensor=x, 
        weights='imagenet'
    )
    base_model.trainable = trainable

    # Output shape : (batch_size, 2048) or (batch_size, 7, 7, 2048)
    return base_model.output

def create_efficientnet_b7_pretrained(input: Input, trainable: bool=False, pooling: bool=True):
    """
    Create pre-trained EfficientNetB7 from keras.applications and return output

    `input.shape` should match (600, 600, 3) 

    Image input should be float32 [0, 255]

    `include_top` is False
    """

    if input.shape[1:] != (600, 600, 3):
        logger.error(f"Invalid Input shape : {input.shape}. It should be (600, 600, 3) for pre-trained EfficientNetB7")

    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_tensor=input,
        pooling='avg' if pooling else None
    )
    base_model.trainable = trainable

    # shape: (batch, 19, 19, 2560) | (batch, 2560)
    return base_model.output

def create_efficientNet_b4_pretrained(input: Input, trainable: bool=False, pooling: bool=True):
    """
    Create pre-trained EfficientNetB7 from keras.applications and return output

    `input.shape` should match (380, 380, 3) 

    Image input should be float32 [0, 255]

    `include_top` is False
    """

    if input.shape[1:] != (380, 380, 3):
        logger.error(f"Invalid Input shape : {input.shape}. It should be (600, 600, 3) for pre-trained EfficientNetB4")

    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=input,
        pooling='avg' if pooling else None
    )
    base_model.trainable = trainable

    # shape: (batch, 12, 12, 1792) | (batch, 1792)
    return base_model.output


def create_resnet152(x, pooling: bool=True):
    """
    Create untrained ResNet152 from scratch.

    `input_shape` should match (224, 224, 3)

    See following link.
    https://github.com/microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/TrainResNet_ImageNet_Distributed.py
    """

    num_filters_list = [64, 128, 256, 512, 1024, 2048]
    block_repeat_list = [2, 7, 35, 2]

    # shape: (batch, 7, 7, 2048)
    x = resnet_bottleneck_model(x, num_filters_list=num_filters_list, block_repeat_list=block_repeat_list)

    if pooling:
        x = tf.keras.layers.GlobalAveragePooling2D()(x) # shape: (batch, 2048)

    return x