#from transformers import TFViTModel
import tensorflow as tf 
from tensorflow.keras import layers, Input, Model

# Using Keras hub for loading pre-trained models
import keras_hub

from src.utils import get_filename_based_logger
from .layers import vit_model, create_custom_vit


logger = get_filename_based_logger(__file__)

def create_vit_224(input: Input, output_cls: bool = False):
    """
    Create untrained ViT feature extractor from scratch

    Structure based on 'google/vit-base-patch16-224'

    Output is (batch, num_patches + 1, projection_dim) when output_cls is False (Default)

    Output is (batch, projection_dim) when output_cls is True
    """

    logger.info(f"Create untrained ViT feature extractor...")

    # Check whether shape is match to ViT
    if input.shape[1:] != (224, 224, 3): # compare without batch
        logger.error(f"Invalid Input shape : {input.shape}. It should be (224, 224, 3) for pretrained ViT")
        return None
    
    output = create_custom_vit(input, output_cls=output_cls)

    logger.info(f"Output shape: {output.shape}")

    return output

def create_vit_base_patch16_224_pretrained(input: Input, trainable: bool=False):
    """
    Create pre-trained ViT feature extractor from 'vit_base_patch16_224_imagenet21k' 
    using 'keras_hub'

    Output is (batch, num_patches + 1, projection_dim)
    """

    logger.info("Create pre-trained ViT feature extractor from 'vit_base_patch16_224_imagenet21k")

    # Check whether shape is match to ViT
    if input.shape[1:] != (224, 224, 3): # compare without batch
        logger.error(f"Invalid Input shape : {input.shape}. It should be (224, 224, 3) for pretrained ViT")
        return None
    
    backbone = keras_hub.models.Backbone.from_preset("vit_base_patch16_224_imagenet21k", trainable=trainable)

    # Replace input layer by using Functional API
    output = backbone(input)

    logger.info(f"Output shape: {output.shape}")

    return output

# Deprecated codes....

# transformers not compatible to Keras 3
""" def create_vit_pretrained(
        input: Input, model_name="google/vit-base-patch16-224", trainable=False
):
    \"""
    Create pre-trained ViT model by using Hugging Face TFVitModel.

    Convert Image to ViT token sequence.
    \"""

    # Check whether shape is match to ViT
    if input.shape[1:] != (224, 224, 3): # compare without batch
        logger.error(f"Invalid Input shape : {input.shape}. It should be (224, 224, 3) for pretrained ViT")
        return None

    # Processing image manually. (without ViTImageProcessor)
    # Resacle [0, 255] to [-1, 1]
    x = layers.Rescaling(1./127.5, offset=-1, name="vit_rescale")(input)

    # Wrap pixel_values for TFViTModel
    def forward_vit(x):
        vit_model = TFViTModel.from_pretrained(model_name)
        vit_model.trainable = trainable
        outputs = vit_model(inputs={"pixel_values" : x})
        return outputs.last_hidden_state 
    
    # Use Lambda layer to compatible for Hugging Face and keras
    # By this, now "vit_model" added to Tensorflow Graph
    vit_tokens = layers.Lambda(forward_vit, name="vit_model")(x)

    return vit_tokens """

""" 
google/vit-base-patch16-224 - ViTImageProcessor config
{
  "do_normalize": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "size": 224
} """
"""
def create_vit_224_pretrained(input: Input, weight_path: str, output_cls: bool = False, trainable: bool=False):
    \"""
    Create pre-trained ViT feature extractor from given weight file.

    Structure based on 'google/vit-base-patch16-224'
    \"""

    # Check whether shape is match to ViT
    if input.shape[1:] != (224, 224, 3): # compare without batch
        logger.error(f"Invalid Input shape : {input.shape}. It should be (224, 224, 3) for pretrained ViT")
        return None
    
    vit_output = create_custom_vit(input, output_cls=output_cls)

    model = Model(inputs=input, outputs=vit_output)

    # Load weights from given weight_path. 
    model.load_weights(weight_path, skip_mismatch=True)

    # Return output tensor
    return model.output
"""