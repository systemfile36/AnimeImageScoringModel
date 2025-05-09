import tensorflow as tf 
from tensorflow.keras import layers

# num_patches = (image_size // patch_size) ** 2
# patch is directly proportional to sequence length and cost of calculation

# Functions to create transformer model from scratch
# https://keras.io/examples/vision/image_classification_with_vision_transformer/

# Layer class must define sub-layer at `__init__`. 
# Because of Keras Layer Compatibility.
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class PatchAndProject(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

        # Define sub-layer in `__init__`

        self.projection = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="vit_patch_embedding"
        )

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        # Auto-infer num_pathes
        x = tf.reshape(x, [batch_size, -1, x.shape[-1]])
        return x #shape =(batch, num_patches, projection_dim)

class AddCLSandPositional(layers.Layer):
    """
    Add positional embedding and CLS token

    Custom Layer class to avoid ValueError.
    Can not use tf.shape() to KerasTensor directly
    """

    def __init__(self, num_patches: int, projection_dim: int, name_prefix: str = "vit", max_len: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.max_len = max_len

        """
        # Positional Embedding: (max_len, projection_dim)
        self.positional_embed = layers.Embedding(
            input_dim=self.max_len,
            output_dim=self.projection_dim,
            name="vit_pos_embedding"
        )
        """

        # Implement positional embedding manually.
        # For match structure to Hugging Face ViT model
        # Can not use tf.Variable in `call`. So use `add_weight`
        self.positional_embed = self.add_weight(
            shape=(1, num_patches + 1, projection_dim),
            initializer="random_normal",
            trainable=True,
            name=f"{name_prefix}_positional_embedding"
        )

        # Add CLS token as trainable
        # Can not use tf.Variable in `call`
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dim),
            initializer="zeros",
            trainable=True,
            name="vit_cls_token"
        )
    def call(self, x):

        batch_size = tf.shape(x)[0]

        # Generate CLS token (trainable vector)
        # Expand CLS token to batch size.
        cls_token_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1]) #shape = (batch, 1, projection_dim)
        x = tf.concat([cls_token_broadcast, x], axis=1) #shape = (batch, num_patches+1, projection_dim)

        # Add positional embedding
        #positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        #x += self.positional_embed(positions)

        # Add positional embedding
        x += self.positional_embed

        return x

class TransformerBlock(layers.Layer):
    def __init__(
            self, projection_dim: int,
            num_heads: int, 
            ffn_dim: int,
            dropout_rate: float = 0.1,
            name_prefix="vit",
            block_index: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        # Calculate `key_dim`
        head_key_dim = projection_dim // num_heads

        self.name_prefix = name_prefix
        self.block_index = block_index

        # Define sub-layers in `__init__`

        # LayerNorm and Dropout, Residual for Attention
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_pre_ln_attention_{block_index}")
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_key_dim,
            name=f"{name_prefix}_attention_{block_index}"
        )
        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{name_prefix}_attention_dropout_{block_index}")
        self.attention_residual = layers.Add(name=f"{name_prefix}_attention_residual_{block_index}")

        # LayerNorm and Dropout, Residual for FFN(MLP)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_pre_ln_ffn_{block_index}")
        self.ffn_1 = layers.Dense(ffn_dim, activation='gelu', name=f"{name_prefix}_ffn_1_{block_index}")
        self.ffn_dropout = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout_{block_index}")
        self.ffn_2 = layers.Dense(projection_dim, name=f"{name_prefix}_ffn_2_{block_index}")
        self.ffn_residual = layers.Add(name=f"{name_prefix}_ffn_residual_{block_index}")

    def call(self, inputs, training=False):

        #Self Attention block
        x = self.layer_norm_1(inputs) # PreNorm
        attention = self.attention(x, x)
        attention = self.attention_dropout(x, training=training)
        x = self.attention_residual([inputs, attention])

        # Feed-Forward block
        ffn = self.layer_norm_2(x) # PreNorm
        ffn = self.ffn_1(ffn)
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_2(ffn)

        # Residual self-attention block and FFN
        x = self.ffn_residual([x, ffn])

        return x

def create_custom_vit(
        input, 
        image_size=224, 
        patch_size=16,
        projection_dim=768,
        num_heads=12,
        ffn_dim=3072,
        num_layers=12,
        dropout_rate=0.1,
        name_prefix="vit_base_p16_224",
        output_cls: bool=True
):
    """
    Create ViT feature extractor from scratch.

    Default value is based on structure of 'google/vit-base-patch16-224'

    
    """
    assert image_size % patch_size == 0, "Image size must be divisible by patch size"
    
    # Normalize to [0, 1]
    x = layers.Rescaling(1./255, name=f"{name_prefix}_rescale_[0,1]")(input)

    # Convert image to patches and flatten
    x = PatchAndProject(patch_size=patch_size, projection_dim=projection_dim)(x)

    # Calculate num_patches
    num_patches = (image_size // patch_size) ** 2

    x = AddCLSandPositional(
        num_patches=num_patches,
        projection_dim=projection_dim,
        name_prefix=name_prefix,
    )(x)

    # Stack Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            projection_dim=projection_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            name_prefix=name_prefix,
            block_index=i
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_final_ln")(x)

    if output_cls:
        # Extract CLS token
        x = layers.Lambda(lambda x: x[:, 0], name=f"{name_prefix}_cls_output")(x)

    return x

# Deprecated codes......

""" def add_positional_and_cls(x, projection_dim):
    \"""
    Add positional embedding and CLS token
    \"""

    # Wrapping to avoid ValueError.
    # Can not use tf.shape() to KerasTensor directly
    def fn(x):
        batch_size = tf.shape(x)[0]
        #num_patches = tf.shape(x)[1]

        # Generate CLS token (trainable vector)
        cls_token = tf.Variable(tf.zeros((1, 1, projection_dim)), name="vit_cls_token") #shape = (1, 1, projection_dim)
        cls_token_broadcast = tf.tile(cls_token, [batch_size, 1, 1]) #shape = (batch, 1, projection_dim)
        x = tf.concat([cls_token_broadcast, x], axis=1) #shape = (batch, num_patches+1, projection_dim)

        # Add positional embedding
        positional_embed = layers.Embedding(input_dim=1000, output_dim=projection_dim, name="vit_pos_embedding")
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        x += positional_embed(positions)

        return x #shape = (batch, num_patch+1, projection_dim)

    return layers.Lambda(fn, name="add_positional_and_cls")(x) """

"""
def vit_model(
        input,
        image_size=224,
        patch_size=16, 
        projection_dim=768,
        transformer_layers=12,
        num_heads=12,
        feed_forward_dim=3072,
        dropout_rate=0.1,
):
    \"""
    Create ViT model from scratch. 

    Default value is ViT-Base (Parameters ~86M)

    Output shape: (batch, num_pathches + 1, projection_dim)
    \"""
    assert image_size % patch_size == 0, "Image size must be divisible by patch size"

    # Normalize to [0, 1]
    x = layers.Rescaling(1./255, name=f"vit_rescale_[0,1]")(input)

    # Convert image to patches and flatten
    x = patchify_images(x, patch_size=patch_size, projection_dim=projection_dim)

    # Add positional embedding and CLS token
    x = AddCLSandPositional(projection_dim=projection_dim)(x)

    # Add Transformer Encoder
    x = transformer_encoder(
        x, 
        projection_dim=projection_dim,
        feed_forward_dim=feed_forward_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layers_count=transformer_layers,
    )

    return x #shape = (batch, num_pathches + 1, projection_dim)
"""

"""
def transformer_encoder(
        x, projection_dim, feed_forward_dim, num_heads, 
        dropout_rate, layers_count, name_prefix="vit"
):
    \"""
    Add Transformer Encoder block.
    \"""

    for i in range(layers_count):
        # Multi-head Attention
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, name=f"{name_prefix}_mhsa_{i}")(x, x)
        attention = layers.Dropout(dropout_rate, name=f"{name_prefix}_attention_dropout_{i}")(attention)
        x = layers.Add(name=f"{name_prefix}_res_attention_{i}")([x, attention])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln_attention_{i}")(x)

        # Feed-forward network
        ffn = layers.Dense(feed_forward_dim, activation="gelu", name=f"{name_prefix}_ffn1_{i}")(x)
        ffn = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout_{i}")(ffn)
        ffn = layers.Dense(projection_dim, name=f"{name_prefix}_ffn2_{i}")(ffn)
        x = layers.Add(name=f"{name_prefix}_res_ffn_{i}")([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln_ffn_{i}")(x)

    return x
"""

"""
def patchify_images(x, patch_size, projection_dim):
    \"""
    Add layer for patchify image.

    Output is flattened patches.
    \"""
    conv = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patchify_conv"
    )(x)

    # num_patches = (image_size // patch_size) ** 2
    patch_dims = (x.shape[1] // patch_size) * (x.shape[2] // patch_size)

    return layers.Reshape((patch_dims, projection_dim), name="patchify_flatten")(conv)
"""