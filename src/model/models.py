import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Callable

from src.utils import get_filename_based_logger
from src.model.layers import TransformerBlock, AddCLSandPositional, AddPositionalOnly
from src.model.layers import data_augmentation

logger = get_filename_based_logger(__file__)

def create_cnn_score_reg_model(
        input: Input, cnn_delegate: Callable[..., any], **kwargs
) -> Model:
    """
    Create simple score regression model based given CNN backbone

    Dataset shape must be 
    
    ```
    (image, 
        {'score_prediction': "int or float32"}
    )
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `cnn_delegate`.
    """

    imgae_feature = cnn_delegate(input, **kwargs)

    # Add regression output.
    score_pred = layers.Dense(1, activation='linear', name='score_prediction')(imgae_feature)

    # Define model
    model = Model(inputs=input, outputs=score_pred)
    
    return model

def create_cnn_score_classification_model(
        input: Input, cnn_delegate: Callable[..., any], 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create simple score classification model based given CNN backbone

    Dataset shape must be 
    
    ```
    (image, 
        {'score_prediction': "int or float32"}
    )
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `cnn_delegate`.
    """

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        image_feature = cnn_delegate(x, **kwargs)
    else:
        image_feature = cnn_delegate(input, **kwargs)

	# Add FFN layer and Dropout
    ffn = layers.Dense(512, activation='relu', name="final_ffn")(image_feature)
    ffn = layers.Dropout(0.3, name="final_dropout")(ffn)

    # Add classification output
    # Make sure output is tf.float32 for mixed precision
    #score_pred = layers.Dense(100, activation="softmax", name='score_prediction', dtype=tf.float32)(ffn)
	
    score_pred_pre = layers.Dense(100)(ffn)

	# Wrapper of clip logits
    # def clip_logits(x):
    #     return tf.clip_by_value(x, -20.0, 20.0)
	
	# # Cliping logits for safety
    # logit_cliping = layers.Lambda(clip_logits)(score_pred_pre)

    score_pred = layers.Softmax(name="score_prediction", dtype=tf.float32)(score_pred_pre)
	
    # Set output name manually to avoid error
    model = Model(inputs=input, outputs={ "score_prediction": score_pred })

    return model

def create_cnn_meta_multitask_reg_model(
        input: Input, cnn_delegate: Callable[..., any], **kwargs
) -> Model:
    """
    Create simple multi-task model with metadata, 'ai_prediction`, `rating_prediction` 
    based given CNN backbone.

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one hot vector", 
        'score_prediction': "float32"
    })
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `cnn_delegate`.
    """

    image_feature = cnn_delegate(input, **kwargs)

    ai_pred = layers.Dense(1, activation='sigmoid', name='ai_prediction')(image_feature)

    rating_pred = layers.Dense(3, activation='softmax', name="rating_prediction")(image_feature)

    score_pred = layers.Dense(1, activation='linear', name='score_prediction')(image_feature)

    model = Model(inputs=input, outputs=[
        ai_pred, rating_pred, score_pred
    ])

    return model

def create_vit_score_reg_by_cls_model(
        input: Input, vit_delegate: Callable[..., any], 
        hidden_dim=256, dropout_rate=0.3, **kwargs
) -> Model:
    """
    Create simple score regression model based given ViT backbone

    Using CLS token

    Dataset shape must be 
    
    ```
    (image, 
        {'score_prediction': "int or float32"}
    )
    ```

    Args:
        input: Input layer of Model
        vit_delegate: Model delegate in `src/model/vit.py`
        hidden_dim: units of final hidden layer
        dropout_rate: Dropout rate of final dropout layer
        **kwargs: args of `vit_delegate`.
    """

    # ViT token
    vit_tokens = vit_delegate(input, **kwargs)

    # Extract CLS token
    cls_token = layers.Lambda(lambda x: x[:, 0], name="cls_token")(vit_tokens)

    # Add Final hidden layer
    x = layers.Dense(hidden_dim, activation='relu', name="final_hidden")(cls_token)
    x = layers.Dropout(dropout_rate, name="final_dropout")(x)

    # Score regression output
    score_pred = layers.Dense(1, activation='linear', name="score_prediction")(x)

    model = Model(inputs=input, outputs=[score_pred])

    return model

def create_vit_meta_multitask_reg_model(
        input: Input, vit_delegate: Callable[..., any], 
        token_dim = 768, final_ff_dim=2048, dropout_rate=0.1, transformer_count=2, 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create multi-task model with metadata, 'ai_prediction`, `rating_prediction` 
    based given ViT backbone.

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one hot vector", 
        'score_prediction': "float32"
    })
    ```

    Args:
        input: Input layer of Model
        vit_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `vit_delegate`.
    """

    # 1. Predict metadata from ViT CLS token.

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        vit_tokens = vit_delegate(x, **kwargs) #shape = (batch, N, dim)
    else:
        vit_tokens = vit_delegate(input, **kwargs) #shape = (batch, N, dim)

    # Extract CLS token for meta predict
    cls_token = layers.Lambda(lambda x: x[:, 0], name="cls_token")(vit_tokens)

    # Predict metadata
    ai_logits = layers.Dense(1, activation="sigmoid", name="ai_prediction")(cls_token)
    rating_logits = layers.Dense(3, activation="softmax", name="rating_prediction")(cls_token)

    # 2. Tokenize predicted metadata and concat with image patches

    # Embedding meta
    ai_token = layers.Dense(token_dim, activation="relu", name="ai_token_proj")(ai_logits)
    rating_token = layers.Dense(token_dim, activation="relu", name="rating_token_proj")(rating_logits)

    # Wrapping tf function to keras.layers.Layer class
    class Stack(layers.Layer):
        def call(self, x):
            return tf.stack(x, axis=1)

    meta_tokens = Stack()([ai_token, rating_token])  #shape = (batch, 2, token_dim)

    # Extract image token and concat it with meta tokens
    patch_tokens = layers.Lambda(lambda x: x[:, 1:], name="exclude_cls")(vit_tokens)
    token_sequence = layers.Concatenate(axis=1, name="concat_tokens")([patch_tokens, meta_tokens])

    # 3. Positional embedding for score prediction

    # # Get Token length 
    # token_sequence_length = token_sequence.shape[1]

    # # Positional embedding for score prediction. 
    # score_pos_embedding = layers.Embedding(input_dim=512, output_dim=token_dim, name="score_pos_embedding")
    # # Get position tensor. length is equal to length of token sequence
    # score_positions = tf.range(start=0, limit=token_sequence_length, delta=1)
    # token_sequence += score_pos_embedding(score_positions)

    token_sequence = AddPositionalOnly(
        num_patches=patch_tokens.shape[1] + meta_tokens.shape[1],
        projection_dim=token_dim,
        name_prefix="score"
    )(token_sequence)

    # 4. Transformer block for score prediction 
    x = token_sequence
    for i in range(transformer_count):
        x = TransformerBlock(
            projection_dim=token_dim, 
            ffn_dim=final_ff_dim,
            num_heads=4, dropout_rate=0.1, block_index=i, name_prefix="score"
        )(x)

    pooled = layers.GlobalAveragePooling1D(name="score_pooling")(x)
    x = layers.Dense(final_ff_dim, activation='relu')(pooled)
    x = layers.Dropout(dropout_rate)(x)
    
    score_pred = layers.Dense(1, activation='linear', name="score_prediction")(x)

    model = Model(inputs=input, outputs=[
        ai_logits, rating_logits, score_pred
    ])

    return model

def create_cnn_meta_multitask_transformer_reg_model(
        input: Input, cnn_delegate: Callable[..., any], 
        token_dim = 768, final_ff_dim=2048, dropout_rate=0.1, transformer_count=2, 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create multi-task model with metadata, 'ai_prediction', 'rating_prediction' 
    based given CNN backbone.

    `cnn_delegate` must return 'feature map'. DONT add pooling layer.

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one hot vector", 
        'score_prediction': "float32"
    })
    ```

    """

    # 1. Convert cnn feature map to token sequence.

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        cnn_feature = cnn_delegate(x, **kwargs)
    else:
        cnn_feature = cnn_delegate(input, **kwargs)

    # Flatten to tokens
    # (batch, H, W, C) -> (batch, H * W, C)
    # auto calculate by `-1`
    patch_tokens = layers.Reshape((-1, cnn_feature.shape[-1]), name="flatten_cnn_tokens")(cnn_feature)

    # Project token to `token_dim`
    # Add activation
    patch_tokens = layers.Dense(token_dim * 2, name="cnn_patch_proj_dense_1")(patch_tokens)
    patch_tokens = layers.Activation("relu", name="cnn_patch_proj_relu")(patch_tokens)
    patch_tokens = layers.Dense(token_dim, name="cnn_patch_proj_dense_2")(patch_tokens)

    # Flatten cnn feature map by GAP for meta prediction
    global_feature = layers.GlobalAveragePooling2D(name="global_feature")(cnn_feature)

    # Predict metadata
    # Make sure output is tf.float32 for mixed precision
    ai_logits = layers.Dense(1, activation="sigmoid", dtype=tf.float32, name="ai_prediction")(global_feature)
    rating_logits = layers.Dense(3, activation="softmax", dtype=tf.float32, name="rating_prediction")(global_feature)

    # Embedding meta
    ai_token = layers.Dense(token_dim, activation="relu", name="ai_token_proj")(ai_logits)
    rating_token = layers.Dense(token_dim, activation="relu", name="rating_token_proj")(rating_logits)
    
    # Wrapping tf function to keras.layers.Layer class
    class Stack(layers.Layer):
        def call(self, x):
            return tf.stack(x, axis=1)

    meta_tokens = Stack()([ai_token, rating_token])  #shape = (batch, 2, token_dim)

    # Concatenate cnn feature token and meta tokens
    token_sequence = layers.Concatenate(axis=1, name="concat_tokens")([patch_tokens, meta_tokens])

    # # Get Token length 
    # token_sequence_length = token_sequence.shape[1]

    # # Positional embedding for score prediction. 
    # score_pos_embedding = layers.Embedding(input_dim=512, output_dim=token_dim, name="score_pos_embedding")
    # # Get position tensor. length is equal to length of token sequence
    # score_positions = tf.range(start=0, limit=token_sequence_length, delta=1)
    # token_sequence += score_pos_embedding(score_positions)

    token_sequence = AddPositionalOnly(
        num_patches=patch_tokens.shape[1] + meta_tokens.shape[1],
        projection_dim=token_dim,
        name_prefix="score"
    )(token_sequence)

    x = token_sequence
    # 4. Transformer block for score prediction 
    for i in range(transformer_count):
        x = TransformerBlock(
            projection_dim=token_dim, 
            ffn_dim=final_ff_dim,
            num_heads=4, dropout_rate=0.1, block_index=i, name_prefix="score"
        )(x)

    # Pooling with GAP1D
    pooled = layers.GlobalAveragePooling1D(name="score_pooling")(x)
    x = layers.Dense(final_ff_dim, activation='relu')(pooled)
    x = layers.Dropout(dropout_rate)(x)
    
    # Make sure output is tf.float32 for mixed precision
    score_pred = layers.Dense(1, activation='linear', dtype=tf.float32, name="score_prediction")(x)

    model = Model(inputs=input, outputs=[
        ai_logits, rating_logits, score_pred
    ])   

    return model

def create_vit_meta_tag_character_multitask_reg_model(
        input: Input, vit_delegate: Callable[..., any], tag_output_dim,
        token_dim = 768, final_ff_dim=2048, dropout_rate=0.1, transformer_count=2, 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create multi-task model with metadata, 'ai_prediction`, `rating_prediction`, 'tag_prediction' 
    based given ViT backbone.

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one hot vector", 
        'score_prediction': "float32", 
        'tag_prediction': "(N,), float32, multi-hot vector
    })
    ```

    Args:
        input: Input layer of Model
        vit_delegate: Model delegate in `src/model/cnn.py`
        tag_output_dim: len(tag_character_all)
        **kwargs: args of `vit_delegate`.
    """

    # 1. Predict metadata from ViT CLS token.

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        vit_tokens = vit_delegate(x, **kwargs) #shape = (batch, N, dim)
    else:
        vit_tokens = vit_delegate(input, **kwargs) #shape = (batch, N, dim)

    # Extract CLS token for meta predict
    cls_token = layers.Lambda(lambda x: x[:, 0], name="cls_token")(vit_tokens)

    # Predict metadata
    # Make sure output is tf.float32 for mixed precision
    ai_logits = layers.Dense(1, activation="sigmoid", dtype=tf.float32, name="ai_prediction")(cls_token)
    rating_logits = layers.Dense(3, activation="softmax", dtype=tf.float32, name="rating_prediction")(cls_token)
    tag_character_logits = layers.Dense(tag_output_dim, activation="sigmoid", dtype=tf.float32, name="tag_prediction")(cls_token)

    # 2. Tokenize predicted metadata and concat with image patches

    # Embedding meta
    ai_token = layers.Dense(token_dim, activation="relu", name="ai_token_proj")(ai_logits)
    rating_token = layers.Dense(token_dim, activation="relu", name="rating_token_proj")(rating_logits)
    tag_character_token = layers.Dense(token_dim, activation="relu", name="tag_character_token_proj")(tag_character_logits)

    # Wrapping tf function to keras.layers.Layer class
    class Stack(layers.Layer):
        def call(self, x):
            return tf.stack(x, axis=1)

    meta_tokens = Stack()([ai_token, rating_token, tag_character_token])  #shape = (batch, 3, token_dim)

    # Extract image token and concat it with meta tokens
    patch_tokens = layers.Lambda(lambda x: x[:, 1:], name="exclude_cls")(vit_tokens)
    token_sequence = layers.Concatenate(axis=1, name="concat_tokens")([patch_tokens, meta_tokens])

    # 3. Positional embedding for score prediction

    # # Get Token length 
    # token_sequence_length = token_sequence.shape[1]

    # # Positional embedding for score prediction. 
    # score_pos_embedding = layers.Embedding(input_dim=512, output_dim=token_dim, name="score_pos_embedding")
    # # Get position tensor. length is equal to length of token sequence
    # score_positions = tf.range(start=0, limit=token_sequence_length, delta=1)
    # token_sequence += score_pos_embedding(score_positions)

    token_sequence = AddPositionalOnly(
        num_patches=patch_tokens.shape[1] + meta_tokens.shape[1],
        projection_dim=token_dim,
        name_prefix="score"
    )(token_sequence)

    # 4. Transformer block for score prediction 
    x = token_sequence
    for i in range(transformer_count):
        x = TransformerBlock(
            projection_dim=token_dim, 
            ffn_dim=final_ff_dim,
            num_heads=4, dropout_rate=0.1, block_index=i, name_prefix="score"
        )(x)

    pooled = layers.GlobalAveragePooling1D(name="score_pooling")(x)
    x = layers.Dense(final_ff_dim, activation='relu')(pooled)
    x = layers.Dropout(dropout_rate)(x)
    
    # Make sure output is tf.float32 for mixed precision
    score_pred = layers.Dense(1, activation='linear', dtype=tf.float32, name="score_prediction")(x)

    model = Model(inputs=input, outputs=[
        ai_logits, rating_logits, tag_character_logits, score_pred
    ])

    return model

def create_cnn_rating_classification_model(
        input: Input, cnn_delegate: Callable[..., any], 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create simple rating classification model based given CNN backbone

    Dataset shape must be 

    ```
    (image, {
        'rating_prediction': "0 or 1, if R-18 then 1 else 0, float32", 
    })
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `vit_delegate`.
    """


    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        image_feature = cnn_delegate(x, **kwargs)
    else:
        image_feature = cnn_delegate(input, **kwargs)

    # Add FFN layer and Dropout
    ffn = layers.Dense(512, activation='relu', name="final_ffn")(image_feature)
    ffn = layers.Dropout(0.3, name="final_dropout")(ffn)

    rating_pred = layers.Dense(1, activation="sigmoid", name="rating_prediction", dtype=tf.float32)(ffn)

    model = Model(inputs=input, outputs={ "rating_prediction": rating_pred })

    return model

def create_cnn_ai_classification_model(
        input: Input, cnn_delegate: Callable[..., any], 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.1,
        **kwargs
) -> Model:
    """
    Create simple AI classification model based given CNN backbone

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "0 or 1, if AI then 1 else 0", 
    })
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `vit_delegate`.
    """

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        image_feature = cnn_delegate(x, **kwargs)
    else:
        image_feature = cnn_delegate(input, **kwargs)

    # Add FFN layer and Dropout
    ffn = layers.Dense(512, activation='relu', name="final_ffn")(image_feature)
    ffn = layers.Dropout(0.3, name="final_dropout")(ffn)

    ai_pred = layers.Dense(1, activation="sigmoid", name="ai_prediction", dtype=tf.float32)(ffn)

    model = Model(inputs=input, outputs={ "ai_prediction": ai_pred })

    return model

def create_cnn_transformer_ai_classification_model(
        input: Input, cnn_delegate: Callable[..., any], 
        token_dim = 768, final_ff_dim=2048, dropout_rate=0.1, transformer_count=2,
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.1,
        **kwargs
) -> Model:
    """
    Create AI classification model based given CNN backbone + Transormer Block

    Dataset shape must be 

    ```
    (image, {
        'ai_prediction': "0 or 1, if AI then 1 else 0", 
    })
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `vit_delegate`.
    """

    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        image_feature = cnn_delegate(x, **kwargs)
    else:
        image_feature = cnn_delegate(input, **kwargs)


    # Flatten to tokens
    # (batch, H, W, C) -> (batch, H * W, C)
    # auto calculate by `-1`
    patch_tokens = layers.Reshape((-1, image_feature.shape[-1]), name="flatten_cnn_tokens")(image_feature)

    # Project token to `token_dim`
    # Add activation
    patch_tokens = layers.Dense(token_dim * 2, name="cnn_patch_proj_dense_1")(patch_tokens)
    patch_tokens = layers.Activation("relu", name="cnn_patch_proj_relu")(patch_tokens)
    
    # (batch, N, C)
    patch_tokens = layers.Dense(token_dim, name="cnn_patch_proj_dense_2")(patch_tokens)

    # Add Positional Encoding (without CLS token)
    token_sequence = AddPositionalOnly(
        num_patches=patch_tokens.shape[1],
        projection_dim=token_dim,
        name_prefix="score"
    )(patch_tokens)

    x = token_sequence

    # Stack TransformerBlock
    for i in range(transformer_count):
        x = TransformerBlock(
            projection_dim=token_dim, 
            ffn_dim=final_ff_dim,
            num_heads=4, dropout_rate=0.1, block_index=i, name_prefix="ai_pred"
        )(x)

    # Pooling with GAP1D
    pooled = layers.GlobalAveragePooling1D(name="score_pooling")(x)
    x = layers.Dense(final_ff_dim, activation='relu')(pooled)
    x = layers.Dropout(dropout_rate)(x)

    ai_pred = layers.Dense(1, activation='sigmoid', dtype=tf.float32, name="ai_prediction")(x)

    model = Model(inputs=input, outputs={ 'ai_prediction': ai_pred })

    return model

def create_vit_rating_classification_model(
        input: Input, vit_delegate: Callable[..., any], 
        augmentation: bool=True, 
        zoom_range: None | float | tuple[float, float] = 0.15,
        rotation_range: None | float | tuple[float, float] = 0.2,
        **kwargs
) -> Model:
    """
    Create simple rating classification model based given ViT backbone

    Dataset shape must be 

    ```
    (image, {
        'rating_prediction': "0 or 1, if R-18 then 1 else 0, float32", 
    })
    ```

    Args:
        input: Input layer of Model
        cnn_delegate: Model delegate in `src/model/cnn.py`
        **kwargs: args of `vit_delegate`.
    """


    # Add augmentation layers when flag set
    if augmentation:
        x = data_augmentation(input, zoom_range=zoom_range, rotation_range=rotation_range)
        vit_tokens = vit_delegate(x, **kwargs)
    else:
        vit_tokens = vit_delegate(input, **kwargs)

    cls_token = layers.Lambda(lambda x: x[:, 0], name="extract_cls_token")(vit_tokens)

    # Add FFN layer and Dropout
    ffn = layers.Dense(512, activation='relu', name="final_ffn")(cls_token)
    ffn = layers.Dropout(0.3, name="final_dropout")(ffn)

    rating_pred = layers.Dense(1, activation="sigmoid", name="rating_prediction", dtype=tf.float32)(ffn)

    model = Model(inputs=input, outputs={ "rating_prediction": rating_pred })

    return model