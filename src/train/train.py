import os
import json
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import src.model.layers

# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import mixed_precision

# Use mixed precision
# https://www.tensorflow.org/guide/mixed_precision?_gl=1*15fhogz*_up*MQ..*_ga*OTA1MzU2MTMxLjE3NDY5ODc0MzA.*_ga_W0YLR4190T*czE3NDY5ODc0MzAkbzEkZzAkdDE3NDY5ODc0MzAkajAkbDAkaDA.
mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TerminateOnNaN

from src.data.dataset import load_all_character_tags, load_records
from src.data.dataset import load_all_character_tags_from_json, save_all_character_tags_as_json
from src.data.dataset_wrappers import DatasetWithMetaAndTagCharacterWrapper, DatasetWithMetaWrapper, DatasetWrapper, DatasetWrapperForScoreClassification, DatasetWrapperForRatingClassification
from src.data.dataset_wrappers import DatasetWrapperForAiClassification
from src.utils import get_filename_based_logger

import src.model.vit
import src.model.cnn
import src.data

from src.model import create_vit_meta_tag_character_multitask_reg_model
from src.model import create_cnn_meta_multitask_transformer_reg_model
from src.model import create_cnn_score_classification_model
from src.model import create_cnn_rating_classification_model
from src.model import create_cnn_ai_classification_model
from src.model import create_cnn_transformer_ai_classification_model

logger = get_filename_based_logger(__file__)

def load_config_from_json(path: str):
    with open(path, "r", encoding="utf-8") as fs:
        return json.load(fs)

def filter_image_exists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a DataFrame to include only rows where the image file exists.
    """

    # bool masking
    exists = df['image_path'].apply(os.path.exists)

    # Logging rows where the image file not exists.
    # Invert 'exists' bool masking array
    removed = df[~exists]

    for value in removed['image_path']:
        logger.debug(f"The records with 'image_path': '{value}' removed.")

    # Filter by mask and reset index
    return df[exists].reset_index(drop=True)

def logging_records(records: pd.DataFrame, *column_names: str):
    logger.debug(f"records keys: {records.to_dict().keys()}")
    
    for column_name in column_names:
        logger.debug(records[column_name])

def save_history_as_json(history, target_path: str):
    """
    Save model.fit() history object to json
    """

    with open(target_path, "w", encoding="utf-8") as fs:
        json.dump(history.history, fs, indent=4)

def load_history_from_json(filepath: str) -> dict:
    """
    Load model.fit() history from json.
    """

    with open(filepath, 'r', encoding="utf-8") as fs:
        return json.load(fs)

def get_project_setting(
        root_path: str, db_path: str, aliases_path: str, project_suffix: str, config_path:str, config_dict: dict
) -> dict:
    
    if not (os.path.exists(root_path) and os.path.isabs(root_path)):
        logger.error(f"root path must be valid absolute path: {root_path}")
        return
    
    # Convert paths to absolute when the path is relative path to root_path

    if not os.path.isabs(db_path):
        db_path = os.path.join(root_path, db_path)

        if not os.path.exists(db_path):
            logger.error(f"invalid db_path: {db_path}")
            return
        
        logger.debug(f"db_path : {db_path}")

    if not os.path.isabs(aliases_path):
        aliases_path = os.path.join(root_path, aliases_path)

        if not os.path.exists(aliases_path):
            logger.error(f"invalid aliases_path: {aliases_path}")
            return

        logger.debug(f"aliases_path: {aliases_path}")

    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(root_path, config_path)

    # Assign config from json or dict
    if config_path and os.path.exists(config_path):
        logger.info(f"Load config from {config_path}")
        # Load config from json
        config = load_config_from_json(config_path)
    elif config_dict:
        logger.info(f"Load config from dict")
        config = config_dict
    else:
        logger.error("invalid config argument")
        return

    # Create project directory under root_path
    project_path = os.path.join(root_path, f"model_project_{project_suffix}")
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    # assign config to value 
    width = config['image_width']
    height = config['image_height']
    learning_rate = config['learning_rate']

    batch_size = config['batch_size']
    epoch_count = config['epoch']

    loss_weights = config['loss_weights'] if 'loss_weights' in config else None

    # Config about data_augmentation
    data_augmentation = config['data_augmentation'] if 'data_augmentation' in config else None

    config_save_path = os.path.join(project_path, "config_used.json")

    model_json_path = os.path.join(project_path, "model_structure.json")

    csv_log_path = os.path.join(project_path, "training_log.csv")

    # Create checkpoint directory
    checkpoint_path = os.path.join(project_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_artifact_path = os.path.join(project_path, f"model-mixed_precision-TF-SavedModel")

    model_path = os.path.join(project_path, f"model-mixed_precision.keras")

    model_history_path = os.path.join(project_path, "model_history.json")

    return {
        'db_path': db_path,
        'aliases_path': aliases_path,
        'config': config,
        'project_path': project_path,
        'width': width,
        'height': height,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epoch_count': epoch_count,
        'loss_weights': loss_weights,
        'data_augmentation': data_augmentation,
        'config_save_path': config_save_path,
        'model_json_path': model_json_path,
        'csv_log_path': csv_log_path,
        'checkpoint_path': checkpoint_path,
        'model_history_path': model_history_path,
        'model_artifact_path': model_artifact_path,
        'model_path': model_path
    }

def train_meta_tag_character_multitask_reg_model(
        root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict):

    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    # debug logging
    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load records from database
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    logger.info(f"Load tag_character list from {aliases_path}")
    tag_character_all = load_all_character_tags(aliases_path)

    # Split train dataset and test dataset
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Caching scaler from train data
    scaler_bookmarks, scaler_views = src.data.preprocessing.get_log_minmax_scaler(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32),
        train_records['total_view'].to_numpy(dtype=np.float32)
    )

    # Score preprocessing with scaler from train dataset
    # Prevent risk of Data Leakage!
    train_records['score_prediction'] = src.data.preprocessing.score_weighted_log_average_time_decay_scaled(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32), 
        train_records['total_view'].to_numpy(dtype=np.float32), 
        train_records['date'].to_numpy(),
        alpha=0.7, 
        scaler_bookmarks=scaler_bookmarks, scaler_views=scaler_views
    ) 

    test_records['score_prediction'] = src.data.preprocessing.score_weighted_log_average_time_decay_scaled(
        test_records['total_bookmarks'].to_numpy(dtype=np.float32), 
        test_records['total_view'].to_numpy(dtype=np.float32), 
        test_records['date'].to_numpy(),
        alpha=0.7, 
        scaler_bookmarks=scaler_bookmarks, scaler_views=scaler_views
    ) 

    # Logging for debug
    logging_records(test_records, 'image_path', 'score_prediction', 'date')

    # Get tf.data.Dataset from subclass of DatasetWrapper (See `src/data/dataset_wrappers.py`)
    train_dataset = DatasetWithMetaAndTagCharacterWrapper(
        train_records, width=width, height=height,
        tag_character_all=tag_character_all,
        normalize=True
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWithMetaAndTagCharacterWrapper(
        test_records, width=width, height=height,
        tag_character_all=tag_character_all,
        normalize=True
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    # Set augmentation flag true when `data_augmentation` config exists
    augmentation: bool = data_augmentation is not None

    model = create_vit_meta_tag_character_multitask_reg_model(
        inputs,
        src.model.vit.create_vit_base_patch16_224_pretrained,
        tag_output_dim=len(tag_character_all),
        token_dim=768, final_ff_dim=2048, dropout_rate=0.1,
        transformer_count=2, 
        augmentation=augmentation, # Set augmentation
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False # freeze pre-trained feature extractor
    )

    model.summary()
    
    # Save model structure
    model_json = model.to_json(indent=2)
    model_json_path = project_setting['model_json_path']
    with open(model_json_path, "w", encoding="utf-8") as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save best model has most low loss.
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "best_model.keras"),
                        monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path)
    ]

    # Compile model with parameters
    model.compile(
        # Use optimizer 'Adam'
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss={
            # tag_prediction is "sigmoid". multi-label classification. (multi-hot vector based)
            "tag_prediction": "binary_crossentropy", 
            # ai_prediction is "sigmoid". binary classification.
            "ai_prediction": "binary_crossentropy",
            # rating_prediction is "softmax". single-label classification. (one-hot vector based)
            "rating_prediction": "categorical_crossentropy",
            # score_prediction is "linear". regression.
            "score_prediction": "mse"
        }, 
        # weights of loss for multi-task model.
        # extract manually to avoid risk of error.
        loss_weights={
            "tag_prediction": loss_weights['tag_prediction'],
            "ai_prediction": loss_weights['ai_prediction'],
            "rating_prediction": loss_weights['rating_prediction'],
            "score_prediction": loss_weights['score_prediction']
        }, 
        metrics={
            # Approximates the AUC (Area under the curve) of the ROC or PR curves. (Default is 'ROC')
            # For evaluate probability based model.
            "tag_prediction": tf.keras.metrics.AUC(name="tag_auc", multi_label=True),
            "ai_prediction": tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
            "rating_prediction": tf.keras.metrics.CategoricalAccuracy(name="rating_acc"),
            "score_prediction": tf.keras.metrics.MeanAbsoluteError(name="score_mae")
        }
    )

    # Start training.
    model_history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epoch_count, 
        callbacks=callbacks, 
        verbose=1
    )

    # Save history
    save_history_as_json(model_history, project_setting['model_history_path'])

    model_path = project_setting['model_path']

    # Save model as keras
    model.save(model_path)

    model_artifact_path = project_setting['model_artifact_path']

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

def train_meta_multitask_reg_model(        
        root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict
):
    
    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load records
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train dataset and test dataset
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Caching scaler from train data
    scaler_bookmarks, scaler_views = src.data.preprocessing.get_log_minmax_scaler(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32),
        train_records['total_view'].to_numpy(dtype=np.float32)
    )

    # Score preprocessing with scaler from train dataset
    # Prevent risk of Data Leakage!
    train_records['score_prediction'] = src.data.preprocessing.score_weighted_log_average_time_decay_scaled(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32), 
        train_records['total_view'].to_numpy(dtype=np.float32), 
        train_records['date'].to_numpy(),
        alpha=0.7, 
        scaler_bookmarks=scaler_bookmarks, scaler_views=scaler_views
    ) 

    test_records['score_prediction'] = src.data.preprocessing.score_weighted_log_average_time_decay_scaled(
        test_records['total_bookmarks'].to_numpy(dtype=np.float32), 
        test_records['total_view'].to_numpy(dtype=np.float32), 
        test_records['date'].to_numpy(),
        alpha=0.7, 
        scaler_bookmarks=scaler_bookmarks, scaler_views=scaler_views
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'score_prediction', 'date')

    # Get tf.data.Dataset from subclass of DatasetWrapper (See `src/data/dataset_wrappers.py`)
    train_dataset = DatasetWithMetaWrapper(
        train_records, width=width, height=height,
        normalize=True
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWithMetaWrapper(
        test_records, width=width, height=height,
        normalize=True
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    # Set augmentation flag true when `data_augmentation` config exists
    augmentation: bool = data_augmentation is not None

    model = create_cnn_meta_multitask_transformer_reg_model(
        inputs, 
        src.model.cnn.create_efficientnet_b7_pretrained, 
        token_dim=768, final_ff_dim=2048, dropout_rate=0.1,
        transformer_count=2,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False, pooling=False
    )

    model.summary()

    model_json = model.to_json(indent=2)
    
    with open(project_setting['model_json_path'], "w", encoding="utf-8") as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save best model has most low loss.
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "best_model.keras"),
                        monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path)
    ]

    model.compile(
        # Use optimizer 'Adam'
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss={
            # ai_prediction is "sigmoid". binary classification.
            "ai_prediction": "binary_crossentropy",
            # rating_prediction is "softmax". single-label classification. (one-hot vector based)
            "rating_prediction": "categorical_crossentropy",
            # score_prediction is "linear". regression.
            "score_prediction": "mse"
        }, 
        # weights of loss for multi-task model.
        # extract manually to avoid risk of error.
        loss_weights={
            "ai_prediction": loss_weights['ai_prediction'],
            "rating_prediction": loss_weights['rating_prediction'],
            "score_prediction": loss_weights['score_prediction']
        }, 
        metrics={
            "ai_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
                tf.keras.metrics.Precision(name="ai_precision"),
                tf.keras.metrics.Recall(name="ai_recall")
            ],
            "rating_prediction": [tf.keras.metrics.CategoricalAccuracy(name="rating_acc")],
            "score_prediction": [tf.keras.metrics.MeanAbsoluteError(name="score_mae")]
        }
    )

    # Start training.
    model_history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epoch_count, 
        callbacks=callbacks, 
        verbose=1
    )

    # Save history
    save_history_as_json(model_history, project_setting['model_history_path'])

    # Save model as keras
    model.save(project_setting['model_path'])

    model_artifact_path = project_setting['model_artifact_path']

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

def train_score_classification_model(
        root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict
):
    
    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    project_path = project_setting['project_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load records
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Caching quantile_transformer from train data
    quantile_bookmarks, quantile_views, quantile_ctr = src.data.preprocessing.get_log_quantile_transformer(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32),
        train_records['total_view'].to_numpy(dtype=np.float32)
    )

    # Score preprocessing with quantile transformer from train dataset
    # Prevent risk of Data Leakage!
    train_records['score_prediction'] = src.data.preprocessing.score_weighted_ctr_log_quantile_time_decay_scaled(
        train_records['total_bookmarks'].to_numpy(dtype=np.float32),
        train_records['total_view'].to_numpy(dtype=np.float32),
        train_records['date'].to_numpy(),
        alpha=0.6, beta=0.2, gamma=0.2,
        qt_bookmarks=quantile_bookmarks,
        qt_views=quantile_views,
        qt_ctr=quantile_ctr, 
        n_quantities=10000,
        time_decay_method="sqrt"
    )

    test_records['score_prediction'] = src.data.preprocessing.score_weighted_ctr_log_quantile_time_decay_scaled(
        test_records['total_bookmarks'].to_numpy(dtype=np.float32),
        test_records['total_view'].to_numpy(dtype=np.float32),
        test_records['date'].to_numpy(),
        alpha=0.6, beta=0.2, gamma=0.2,
        qt_bookmarks=quantile_bookmarks, # reuse QuantileTransformer
        qt_views=quantile_views, # reuse QuantileTransformer
        qt_ctr=quantile_ctr, 
        n_quantities=10000,
        time_decay_method="sqrt", 
        time_decay_lambda=0.2
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'score_prediction')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    # Get tf.data.Dataset
    train_dataset = DatasetWrapperForScoreClassification(
        train_records, width=width, height=height,
        num_classes = 100, normalize=True
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForScoreClassification(
        test_records, width=width, height=height,
        num_classes = 100, normalize=True
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_score_classification_model(
        inputs, 
        #src.model.cnn.create_resnet152,
        src.model.cnn.create_resnet152_pretrained, # Use pre-trained for fine-tuning
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False, # Freeze feature extractor for fine-tuning
        pooling=True # pooling is true because output is flat dense
    )

    model.summary()

    model_json = model.to_json(indent=2)

    with open(project_setting['model_json_path'], "w", encoding='utf-8') as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    model.compile(
        # Use optimizer 'Adam'
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss={
            # Use 'categorical_crossentropy' for stable train
            "score_prediction": "categorical_crossentropy"
        }, 
        metrics={
            # Use custom MAE function for metric
            "score_prediction": [expected_mae]
        }
    )

    model_history = model.fit(
        train_dataset, 
        validation_data=test_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
        verbose=1
    )

    save_history_as_json(model_history, project_setting['model_history_path'])

    # Save model as keras
    model.save(project_setting['model_path'])

    model_artifact_path = project_setting['model_artifact_path']

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

    fine_tuning_pre_trained_based_model(
        project_setting=project_setting,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss={
            # Use 'categorical_crossentropy' for stable train
            "score_prediction": "categorical_crossentropy"
        },
        metrics={
            # Use custom MAE function for metric
            "score_prediction": [expected_mae]
        },
        epoch=10,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="conv5_block2",
        new_learning_rate=1e-5
    )

def fine_tuning_pre_trained_based_model(
        project_setting, model: Model, 
        train_dataset, test_dataset,
        loss, metrics, epoch, 
        fine_tune_suffix="fine_tune_0",
        unfreeze_boundary_name="conv5_block2",
        new_learning_rate = 1e-5):
    """
    Second step of fine-tuning pre-trained model. 

    Use same project_setting and Model instance, loss, metrics to first step.

    You can specify learning_rate. Default is 1e-5
    """

    # Unfreeze all layers
    model.trainable = True

    # Freeze all layers before the layer contain `unfreeze_boundary_name` in name
    set_trainable = False
    for layer in model.layers:
        if unfreeze_boundary_name in layer.name:
            set_trainable = True

        layer.trainable = set_trainable

    checkpoint_path = os.path.join(project_setting['project_path'], f"checkpoint_{fine_tune_suffix}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    csv_log_path = os.path.join(project_setting['project_path'], f"training_log_{fine_tune_suffix}.csv")

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

                # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),

        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    # Compile again with new leaning rate and unfreeze model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate),
        loss=loss,
        metrics=metrics
    )

    model_history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epoch,
        callbacks=callbacks,
        verbose=1
    )

    # Save fine-tuned model
    model_path = os.path.join(project_setting['project_path'], f"model-mixed_precision_{fine_tune_suffix}.keras")

    model.save(model_path)

    model_artifact_path = os.path.join(project_setting['project_path'], f"model-mixed_precision-TF-SavedModel_{fine_tune_suffix}")

    model.export(model_artifact_path)

def train_rating_classification_model(
        root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict    
):
    
    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    project_path = project_setting['project_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load records
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'rating_prediction')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForRatingClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForRatingClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_rating_classification_model(
        inputs,
        src.model.cnn.create_efficientnet_b7_pretrained,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False,
        pooling=True
    )

    model.summary()

    model_json = model.to_json(indent=2)

    with open(project_setting['model_json_path'], "w", encoding='utf-8') as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    model.compile(
        # Use optimizer 'Adam'
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        # Set loss and metrics for binary classification
        loss={
            "rating_prediction": "binary_crossentropy"
        }, 
        metrics={
            "rating_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="rating_acc"),
                tf.keras.metrics.Precision(name="rating_precision"),
                tf.keras.metrics.Recall(name="rating_recall")
            ]
        }
    )

    model_history = model.fit(
        train_dataset, 
        validation_data=test_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
        verbose=1
    )

    save_history_as_json(model_history, project_setting['model_history_path'])

    # Save model as keras
    model.save(project_setting['model_path'])

    model_artifact_path = project_setting['model_artifact_path'] 

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

    fine_tuning_pre_trained_based_model(
        project_setting=project_setting,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # Set loss and metrics for binary classification
        loss={
            "rating_prediction": "binary_crossentropy"
        }, 
        metrics={
            "rating_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="rating_acc"),
                tf.keras.metrics.Precision(name="rating_precision"),
                tf.keras.metrics.Recall(name="rating_recall")
            ]
        },
        epoch=10,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5
    )

def train_ai_classification_model(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict    
):
    
    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    project_path = project_setting['project_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

        logger.info(f"Load records from {db_path}")

    # Load records
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'ai_prediction')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForAiClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForAiClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_ai_classification_model(
        inputs,
        src.model.cnn.create_efficientNet_b4_pretrained,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False,
        pooling=True
    )

    model.summary()

    model_json = model.to_json(indent=2)

    with open(project_setting['model_json_path'], "w", encoding='utf-8') as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    model.compile(
        # Use optimizer 'Adam'
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        # Set loss and metrics for binary classification
        loss={
            "ai_prediction": "binary_crossentropy"
        }, 
        metrics={
            "ai_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
                tf.keras.metrics.Precision(name="ai_precision"),
                tf.keras.metrics.Recall(name="ai_recall")
            ]
        }
    )

    model_history = model.fit(
        train_dataset, 
        validation_data=test_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
        verbose=1
    )

    save_history_as_json(model_history, project_setting['model_history_path'])

    # Save model as keras
    model.save(project_setting['model_path'])

    model_artifact_path = project_setting['model_artifact_path'] 

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

    fine_tuning_pre_trained_based_model(
        project_setting=project_setting,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # Set loss and metrics for binary classification
        loss={
            "ai_prediction": "binary_crossentropy"
        }, 
        metrics={
            "ai_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
                tf.keras.metrics.Precision(name="ai_precision"),
                tf.keras.metrics.Recall(name="ai_recall")
            ]
        },
        epoch=10,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5
    )

def train_ai_classification_transformer_model(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict    
):
    
    project_setting = get_project_setting(root_path, db_path, aliases_path, project_suffix, config_path, config_dict)

    config = project_setting['config']
    db_path = project_setting['db_path']
    aliases_path = project_setting['aliases_path']
    project_path = project_setting['project_path']
    width = project_setting['width']
    height = project_setting['height']
    learning_rate = project_setting['learning_rate']
    batch_size = project_setting['batch_size']
    epoch_count = project_setting['epoch_count']
    loss_weights = project_setting['loss_weights']
    data_augmentation = project_setting['data_augmentation']

    logger.debug(f"Config set : {config}")

    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

        logger.info(f"Load records from {db_path}")

    # Load records
    records = load_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'ai_prediction')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForAiClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForAiClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_transformer_ai_classification_model(
        inputs,
        src.model.cnn.create_efficientNet_b4_pretrained,
        token_dim=768, final_ff_dim=2048, dropout_rate=0.1,
        transformer_count=2,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False,
        pooling=True
    )

    model.summary()

    model_json = model.to_json(indent=2)

    with open(project_setting['model_json_path'], "w", encoding='utf-8') as fs:
        fs.write(model_json)

    csv_log_path = project_setting['csv_log_path']

    # Create checkpoint directory
    checkpoint_path = project_setting['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

        # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    # Get steps per epoch (length of dataset / batch size)
    steps_per_epoch = int(len(train_records['image_path']) / batch_size)

    # Calculate count of total steps
    total_steps = epoch_count * steps_per_epoch

    # Set warm-up steps. 10% of total steps
    warmup_steps = int(0.1 * total_steps)

    # Set LearningRateSchedule
    learning_rate_schedule = WarmUpCosineDecay(
        base_lr=learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )

    logger.debug(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}")

    model.compile(
        # Use optimizer 'AdamW' for Transformer
        # Set weight decay to avoid overfitting
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_schedule,
            weight_decay=1e-4), 
        # Set loss and metrics for binary classification
        loss={
            "ai_prediction": "binary_crossentropy"
        }, 
        metrics={
            "ai_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
                tf.keras.metrics.Precision(name="ai_precision"),
                tf.keras.metrics.Recall(name="ai_recall")
            ]
        }
    )

    model_history = model.fit(
        train_dataset, 
        validation_data=test_dataset,
        epochs=epoch_count,
        callbacks=callbacks,
        verbose=1
    )

    save_history_as_json(model_history, project_setting['model_history_path'])

    # Save model as keras
    model.save(project_setting['model_path'])

    model_artifact_path = project_setting['model_artifact_path'] 

    # Save model as TF SavedModel
    # See following link. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
    model.export(model_artifact_path)

    # Using AdamW too in fine tuning 
    fine_tuning_pre_trained_based_model_with_warmup(
        project_setting=project_setting,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # Set loss and metrics for binary classification
        loss={
            "ai_prediction": "binary_crossentropy"
        }, 
        metrics={
            "ai_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="ai_acc"),
                tf.keras.metrics.Precision(name="ai_precision"),
                tf.keras.metrics.Recall(name="ai_recall")
            ]
        },
        epoch=10,
        total_steps=total_steps, warmup_steps=warmup_steps,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5
    )

def expected_mae(y_true, y_pred):
    """
    Compute MAE from expected score of softmax output
    """
    # y_pred is softmax, y_true is soft label
  	# Replace NaNs to 0 for safety 
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred, dtype=tf.float32), y_pred)
 
    # Get length of softmax output vector
    num_classes = tf.shape(y_pred)[-1]

    # Get Class index tensor like [1.0, 2.0, 3.0, 4.0, ......]
    class_indices = tf.cast(tf.range(1, num_classes + 1), tf.float32)

    true_score = tf.reduce_sum(y_true * class_indices, axis=-1)
    pred_score = tf.reduce_sum(y_pred * class_indices, axis=-1)

    # Compute MAE
    return tf.reduce_mean(tf.abs(true_score - pred_score))

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom LearningRateSchedule class

    Linear Warm-up + Cosine decay for train TransformerBlock with AdamW 
    """
    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear Warm-up
        warmup_lr = self.base_lr * (step / self.warmup_steps)

        # Cosine Decay after warm-up
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(progress, 1.0)))
        decayed = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr

        # Return warmup_lr if step less than warmup_steps else decayed
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed)

def fine_tuning_pre_trained_based_model_with_warmup(
        project_setting, model: Model, 
        train_dataset, test_dataset,
        loss, metrics, epoch, 
        total_steps, warmup_steps,
        fine_tune_suffix="fine_tune_0",
        unfreeze_boundary_name="conv5_block2",
        new_learning_rate = 1e-5):
    """
    Second step of fine-tuning pre-trained model. 

    Use same project_setting and Model instance, loss, metrics with first step.

    You can specify learning_rate. Default is 1e-5

    Using AdamW + Warm-up + Cosine Decay
    """

    # Unfreeze all layers
    model.trainable = True

    # Freeze all layers before the layer contain `unfreeze_boundary_name` in name
    set_trainable = False
    for layer in model.layers:
        if unfreeze_boundary_name in layer.name:
            set_trainable = True

        layer.trainable = set_trainable

    checkpoint_path = os.path.join(project_setting['project_path'], f"checkpoint_{fine_tune_suffix}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    csv_log_path = os.path.join(project_setting['project_path'], f"training_log_{fine_tune_suffix}.csv")

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1), 

                # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),

        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    # Set LearningRateSchedule
    learning_rate_schedule = WarmUpCosineDecay(
        base_lr=new_learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )

    logger.debug(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}")

    # Compile again with new leaning rate and unfreeze model
    model.compile(
        # Set Weight decay to 1e-4 to avoid overfitting
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_schedule, weight_decay=1e-4),
        loss=loss,
        metrics=metrics
    )

    model_history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epoch,
        callbacks=callbacks,
        verbose=1
    )

    # Save fine-tuned model
    model_path = os.path.join(project_setting['project_path'], f"model-mixed_precision_{fine_tune_suffix}.keras")

    model.save(model_path)

    model_artifact_path = os.path.join(project_setting['project_path'], f"model-mixed_precision-TF-SavedModel_{fine_tune_suffix}")

    model.export(model_artifact_path)


def execution_example():
    """
    Example of running train function. 

    For debugging and suggest example
    """

    train_meta_tag_character_multitask_reg_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_single_page.sqlite3", 
        ".database/character_tags_post500_aliases.sqlite3", '1st_experiment_debug', None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 0.0003,
            'batch_size': 64, 'epoch': 20, 'loss_weights': {
                'score_prediction': 0.01, 'ai_prediction': 0.8, 'rating_prediction': 1.0, 
                'tag_prediction': 1.0
            }, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_example_v2():
    """
    Function for experiment
    """

    train_meta_multitask_reg_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag.sqlite3", 
        "", '3rd_experiment_efnb7_based', None, {
            'image_width': 600, 'image_height': 600, 'learning_rate': 0.0003,
            'batch_size': 64, 'epoch': 28, 'loss_weights': {
                'score_prediction': 0.01, 'ai_prediction': 1.0, 'rating_prediction': 1.0,
            }, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def excution_example_v3():
    """
    Function for experiment
    """

    train_score_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_only_original.sqlite3",
        "", "4th_experiment_cnn_based_score_classification", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 0.001,
            'batch_size': 32, 'epoch': 28, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_example_v4():
    """
    Function for experiment '4th_experiment_pre-trained_resnet_based_score_classification'
    """

    train_score_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_only_original_without_r-18_ai.sqlite3",
        "", "4th_experiment_pre-trained_resnet_based_score_classification", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 0.001,
            'batch_size': 32, 'epoch': 28, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_example_v5():
    """
    Function for experiment '5th_experiment_pre-trained_efficientnetb7_based_rating_classification'
    """

    train_rating_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_r-18_sampling.sqlite3", "",
        "5th_experiment_pre-trained_efficientnetb7_based_rating_classification", None, {
            'image_width': 600, 'image_height': 600, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def fine_tuning_example_v5():
    """
    For Fine-tuning experiment '5th_experiment_pre-trained_efficientnetb7_based_rating_classification'
    
    Reduce `batch_size` because OOM error has occured in `train_rating_classification_model`
    """
    
    logger.info(f"find-tuning `5th_experiment_pre-trained_efficientnetb7_based_rating_classification`")

    # Load trained model
    model = tf.keras.models.load_model("/data/PixivDataBookmarks/model_project_5th_experiment_pre-trained_efficientnetb7_based_rating_classification/model-mixed_precision.keras")

    # Same as example_v5
    project_setting = get_project_setting(
        "/data/PixivDataBookmarks", ".database/metadata_base_r-18_sampling.sqlite3", "",
        "5th_experiment_pre-trained_efficientnetb7_based_rating_classification", None, {
            'image_width': 600, 'image_height': 600, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

    # Reload dataset
    # Same step as example_v5

    logger.info(f"Load records from {project_setting['db_path']}")

    # Load records
    records = load_records("/data/PixivDataBookmarks", ".database/metadata_base_r-18_sampling.sqlite3")

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'rating_prediction')

    width = project_setting['width']
    height = project_setting['height']

    batch_size = 16 # reduce batch_size to avoid OOM error

    train_dataset = DatasetWrapperForRatingClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForRatingClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    fine_tuning_pre_trained_based_model(
        project_setting=project_setting,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # Set loss and metrics for binary classification
        loss={
            "rating_prediction": "binary_crossentropy"
        }, 
        metrics={
            "rating_prediction": [
                tf.keras.metrics.BinaryAccuracy(name="rating_acc"),
                tf.keras.metrics.Precision(name="rating_precision"),
                tf.keras.metrics.Recall(name="rating_recall")
            ]
        },
        epoch=10,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5 # reduce learning_rate for find-tuning
    )

def execution_example_v6():
    """
    Function for experiment '6th_experiment_pre-trained_efficientnetb4_based_ai_classification`
    """
    
    train_ai_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag.sqlite3", 
        "", "6th_experiment_pre-trained_efficientnetb4_based_ai_classification", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.1
            }
        }
    )

def execution_example_v6_1():
    """
    Function for experiment '6th_experiment_pre-trained_efficientnetb4_transformer_ai_classification`
    """
    
    train_ai_classification_transformer_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag.sqlite3", 
        "", "6th_experiment_pre-trained_efficientnetb4_transformer_ai_classification", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.0001,
            'batch_size': 48, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.1
            }
        }
    )

def for_test():

    print(get_project_setting(
        "/data/PixivDataBookmarks", ".database/metadata_base_single_page.sqlite3", 
        ".database/character_tags_post500_aliases.sqlite3", 'for_test_temporary', None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 0.0003,
            'batch_size': 64, 'epoch': 20, 'loss_weights': {
                'score_prediction': 0.01, 'ai_prediction': 0.8, 'rating_prediction': 1.0, 
                'tag_prediction': 1.0
            }, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    ))

if __name__ == "__main__":
    # For Test

    # execution_example()
    # execution_example_v2()
    # excution_example_v3()
    # execution_example_v4()
    # # execution_example_v5()
    # fine_tuning_example_v5()
    # execution_example_v6()
    execution_example_v6_1()