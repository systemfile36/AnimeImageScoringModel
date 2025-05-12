import os
import json
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from src.data.dataset import load_all_character_tags, load_records
from src.data.dataset import load_all_character_tags_from_json, save_all_character_tags_as_json
from src.data.dataset_wrappers import DatasetWithMetaAndTagCharacterWrapper, DatasetWithMetaWrapper, DatasetWrapper
from src.utils import get_filename_based_logger

import src.model.vit
import src.model.cnn
import src.data

from src.model import create_vit_meta_tag_character_multitask_reg_model
from src.model import create_cnn_meta_multitask_transformer_reg_model

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

    loss_weights = config['loss_weights']

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

    model_json_path = os.path.join(project_path, "model_history.json")

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
    execution_example_v2()