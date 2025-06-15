import os
import json
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import KFold

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

from src.data.dataset import load_all_character_tags, load_records, load_score_records, load_quality_binary_records
from src.data.dataset import load_all_character_tags_from_json, save_all_character_tags_as_json
from src.data.dataset_wrappers import DatasetWithMetaAndTagCharacterWrapper, DatasetWithMetaWrapper, DatasetWrapper, DatasetWrapperForScoreClassification, DatasetWrapperForRatingClassification
from src.data.dataset_wrappers import DatasetWrapperForAiClassification
from src.data.dataset_wrappers import DatasetWrapperForManualScoreClassification
from src.data.dataset_wrappers import DatasetWrapperForManualScoreClassificationWithMultiTask
from src.data.dataset_wrappers import DatasetWrapperForManualScoreRegressionWithMultiTask
from src.data.dataset_wrappers import DatasetWrapperForSanityLevelClassification
from src.data.dataset_wrappers import DatasetWrapperForAestheticBinaryClassification

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
from src.model import create_cnn_manual_score_classification_model
from src.model import create_cnn_manual_score_classification_model_v2
from src.model import create_cnn_manual_score_classification_model_v3
from src.model import create_cnn_manual_score_classification_model_v4
from src.model import create_cnn_manual_score_classification_model_v5
from src.model import create_cnn_manual_score_classification_model_v6
from src.model import create_cnn_manual_score_classification_multi_task_model_v1
from src.model import create_cnn_manual_score_regression_multi_task_model_v1
from src.model import create_cnn_transformer_quality_binary_classification_model
from src.model import create_cnn_quality_binary_classification_model
from src.model import create_vit_quality_binary_classification_model

from src.model import create_cnn_sanity_level_classification_model
from src.model import create_cnn_transformer_sanity_level_classification_model

from src.metrics import MacroPrecision, MacroRecall

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
        loss_weights = None,
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

        # Exclude BatchNormalization. stay freeze
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    checkpoint_path = os.path.join(project_setting['project_path'], f"checkpoint_{fine_tune_suffix}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    csv_log_path = os.path.join(project_setting['project_path'], f"training_log_{fine_tune_suffix}.csv")

    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 5 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1), 

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
        loss_weights=loss_weights,
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
        normalize=False # normalize false for pre-trained EfficientNetB0
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForRatingClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB0
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_rating_classification_model(
        inputs,
        # src.model.cnn.create_efficientnet_b7_pretrained,
        src.model.cnn.create_efficientNet_b0_pretrained, # backbone change
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
        epoch=15,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5
    )

def train_sanity_level_classification_model(
        root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict    
        , gaussian_sigma: float = 0.7
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
    logging_records(test_records, 'image_path', 'sanity_level')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForSanityLevelClassification(
        train_records, width=width, height=height,
        normalize=False, # normalize false for pre-trained EfficientNet
        gaussian_sigma=gaussian_sigma
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForSanityLevelClassification(
        test_records, width=width, height=height,
        normalize=False, # normalize false for pre-trained EfficientNet
        gaussian_sigma=gaussian_sigma
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_sanity_level_classification_model(
        inputs,
        src.model.cnn.create_efficientNet_b4_pretrained,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False,
        pooling=True # add avg pooling for MLP head
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
        # Set loss and metrics for Gaussian soft-label
        loss={
            "sanity_level": tf.keras.losses.KLDivergence()
        }, 
        metrics={
            "sanity_level": [
                expected_mae_for_sanity,
                MacroRecall(num_classes=4),
                MacroPrecision(num_classes=4)
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
        # Set loss and metrics for Gaussian soft-label
        loss={
            "sanity_level": tf.keras.losses.KLDivergence()
        }, 
        metrics={
            "sanity_level": [
                expected_mae_for_sanity,
                MacroRecall(num_classes=4),
                MacroPrecision(num_classes=4)
            ]
        },
        epoch=15,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block6a", # freeze after block6a_expand_conv
        new_learning_rate=1e-5
    )

def train_sanity_level_classification_transformer_model(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict,    
    gaussian_sigma: float = 0.7
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
    logging_records(test_records, 'image_path', 'sanity_level')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForSanityLevelClassification(
        train_records, width=width, height=height,
        normalize=False, # normalize false for pre-trained EfficientNet
        gaussian_sigma=gaussian_sigma
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForSanityLevelClassification(
        test_records, width=width, height=height,
        normalize=False, # normalize false for pre-trained EfficientNet
        gaussian_sigma=gaussian_sigma
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_transformer_sanity_level_classification_model(
        inputs,
        src.model.cnn.create_efficientNet_b4_pretrained,
        token_dim=768, final_ff_dim=2048, dropout_rate=0.1, 
        transformer_count=2,
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        trainable=False,
        pooling=False # pooling false for Transformer block
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
        # Monitor loss. stop training when no improvement in 7 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1), 

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
        # Set loss and metrics for Gaussian soft-label
        loss={
            "sanity_level": tf.keras.losses.KLDivergence()
        }, 
        metrics={
            "sanity_level": [
                expected_mae_for_sanity,
                MacroRecall(num_classes=4),
                MacroPrecision(num_classes=4)
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
        # Set loss and metrics for Gaussian soft-label
        loss={
            "sanity_level": tf.keras.losses.KLDivergence()
        }, 
        metrics={
            "sanity_level": [
                expected_mae_for_sanity,
                MacroRecall(num_classes=4),
                MacroPrecision(num_classes=4)
            ]
        },
        epoch=15,
        total_steps=total_steps, warmup_steps=warmup_steps,
        steps_per_epoch=steps_per_epoch,
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

def train_pseudo_label_generator_by_manual_score(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict
):
    
    logger.info(f"Train {project_suffix} start...")

    # extract setting to dictionary
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

    # Save config file
    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load score records
    records = load_score_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'manual_score')

    # Export test records for debug and statistics
    test_records.to_csv(os.path.join(project_path, "test_records.csv"))

    train_dataset = DatasetWrapperForManualScoreClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForManualScoreClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB7
    ).get_dataset(batch_size=batch_size)

    # Create Input layer
    inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

    augmentation: bool = data_augmentation is not None

    model = create_cnn_manual_score_classification_model(
        inputs, 
        src.model.cnn.create_efficientNet_b4_pretrained, 
        augmentation=augmentation,
        zoom_range=data_augmentation['zoom_range'],
        rotation_range=data_augmentation['rotation_range'],
        ffn_dim=512, dropout_rate=0.3,
        num_classes=8,
        trainable=False,
        pooling=True # add 'avg' pooling for MLP head
    )

    model.summary()

    # Save model structure for debugging 
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
        # Set loss and metrics for soft-label based classification
        loss={
            "manual_score": "categorical_crossentropy"
        }, 
        metrics={
            "manual_score": [expected_mae]
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
        # Use same loss and metrics with train
        loss={
            "manual_score": "categorical_crossentropy"
        }, 
        metrics={
            "manual_score": [expected_mae]
        },
        epoch=20,
        fine_tune_suffix="fine_tune_1",
        unfreeze_boundary_name="block_6a",
        new_learning_rate=1e-5
    )

def train_pseudo_label_generator_by_manual_score_k_fold(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict, 
    gaussian_sigma: float = 1.0
):
    """
    Train CNN + MLP based pseudo-lable generator from manual labeled score. 

    Using K-Fold Cross Validation
    """

    logger.info(f"Train {project_suffix} start...")

    # extract setting to dictionary
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

    # Save config file
    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load score records
    records = load_score_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Generate K-Fold index array
    # This will generate 
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    logger.info("Start K-Fold iteration")

    # K-Fold iteration
    for fold, (train_i, test_i) in enumerate(kfold.split(records)):
        logger.info(f"Start Fold {fold + 1}==========================")

        # Create sub directory each fold
        fold_project_path = os.path.join(project_path, f"fold_{fold + 1}")
        if not os.path.exists(fold_project_path):
            os.makedirs(fold_project_path)

        # Split train and test by K-Fold index array
        train_records = records.iloc[train_i]
        test_records = records.iloc[test_i]

        # Logging for debug
        logging_records(test_records, 'image_path', 'manual_score')

        # Export test records for debug and statistics
        test_records.to_csv(os.path.join(fold_project_path, "test_records.csv"))

        train_dataset = DatasetWrapperForManualScoreClassification(
            train_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
            sigma=gaussian_sigma, 
            num_classes=3 # Coarse 3-class soft-label
        ).get_dataset(batch_size=batch_size)

        test_dataset = DatasetWrapperForManualScoreClassification(
            test_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
            sigma=gaussian_sigma,
            num_classes=3 # Coarse 3-class soft-label
        ).get_dataset(batch_size=batch_size)

        # Create Input layer
        inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

        augmentation: bool = data_augmentation is not None

        # model = create_cnn_manual_score_classification_model_v2(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b1_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model_v3(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b4_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model_v4(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b0_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model_v5(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b4_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     trainable=False,
        #     # pooling=True # add 'avg' pooling for MLP head
        #     pooling=False # set pooling to false for self-attention
        # )

        # model = create_cnn_manual_score_classification_model_v6(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b4_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model_v4(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b0_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=3, # Coarse 3-class soft-label
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model(
        #     inputs, 
        #     src.model.cnn.create_efficientNet_b0_pretrained, 
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=3, # Coarse 3-class soft-label
        #     ffn_dim=256, # Reduce model capacity
        #     dropout_rate=0.3,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        # model = create_cnn_manual_score_classification_model(
        #     inputs, 
        #     src.model.cnn.create_mobilenet_v2_pretrained, # change backbone
        #     augmentation=augmentation,
        #     zoom_range=data_augmentation['zoom_range'],
        #     rotation_range=data_augmentation['rotation_range'],
        #     num_classes=8,
        #     ffn_dim=256, # Reduce model capacity
        #     dropout_rate=0.3,
        #     trainable=False,
        #     pooling=True # add 'avg' pooling for MLP head
        # )

        model = create_cnn_manual_score_classification_model(
            inputs, 
            src.model.cnn.create_mobilenet_v2_pretrained, # change backbone
            augmentation=augmentation,
            zoom_range=data_augmentation['zoom_range'],
            rotation_range=data_augmentation['rotation_range'],
            num_classes=3, # Coarse 3-class soft-label
            ffn_dim=256, # Reduce model capacity
            dropout_rate=0.3,
            trainable=False,
            pooling=True # add 'avg' pooling for MLP head
        )

        # Save model structure for debugging 
        model_json = model.to_json(indent=2)
        model_json_path = os.path.join(fold_project_path, "model_structure.json")
        with open(model_json_path, "w", encoding='utf-8') as fs:
            fs.write(model_json)

        csv_log_path = os.path.join(fold_project_path, "training_log.csv")

        # Create checkpoint directory
        checkpoint_path = os.path.join(fold_project_path, "checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        callbacks = [
            # Set EarlyStopping. 
            # Monitor loss. stop training when no improvement in 5 epochs.
            EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1), 

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
            # Use KLDivergence for softmax and soft-label
            loss={
                "manual_score": tf.keras.losses.KLDivergence()
            }, 
            metrics={
                "manual_score": [expected_mae]
            }
        )

        model_history = model.fit(
            train_dataset, 
            validation_data=test_dataset,
            epochs=epoch_count,
            callbacks=callbacks,
            verbose=1
        )

        model_history_path = os.path.join(fold_project_path, "model_history.json")
        save_history_as_json(model_history, model_history_path)

        # Save model as keras each fold
        model_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}.keras")
        model.save(model_path)

        model_artifact_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}-TF-SavedModel")

        # Save model as TF SavedModel
        # See following link. 
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
        model.export(model_artifact_path)

        project_setting['project_path'] = fold_project_path

        fine_tuning_pre_trained_based_model(
            project_setting=project_setting,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            # Use KLDivergence for softmax and soft-label
            loss={
                "manual_score": tf.keras.losses.KLDivergence()
            }, 
            metrics={
                "manual_score": [expected_mae]
            },
            epoch=20,
            fine_tune_suffix=f"fine_tune_1_fold{fold + 1}",
            # unfreeze_boundary_name="block6a",
            # unfreeze_boundary_name="block7a", # reduce unfreeze block count
            unfreeze_boundary_name="block_13", # For MobileNetV2
            new_learning_rate=1e-5
            # new_learning_rate=5e-6
        )

def train_pseudo_label_generator_by_manual_score_multi_task_k_fold(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict, 
    gaussian_sigma: float = 1.0
):
    """
    Train CNN + MLP based pseudo-lable generator from manual labeled score. 

    Predict multi-task with `color_lighting_score` and `costume_detail_score`, `proportion_score`
    
    Using K-Fold Cross Validation
    """

    logger.info(f"Train {project_suffix} start...")

    # extract setting to dictionary
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

    # Save config file
    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load score records
    records = load_score_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Generate K-Fold index array
    # This will generate 
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    logger.info("Start K-Fold iteration")

    # K-Fold iteration
    for fold, (train_i, test_i) in enumerate(kfold.split(records)):
        logger.info(f"Start Fold {fold + 1}==========================")

        # Create sub directory each fold
        fold_project_path = os.path.join(project_path, f"fold_{fold + 1}")
        if not os.path.exists(fold_project_path):
            os.makedirs(fold_project_path)

        # Split train and test by K-Fold index array
        train_records = records.iloc[train_i]
        test_records = records.iloc[test_i]

        # Logging for debug
        logging_records(test_records, 'image_path', 'manual_score', 'color_lighting_score', 'costume_detail_score')

        # Export test records for debug and statistics
        test_records.to_csv(os.path.join(fold_project_path, "test_records.csv"))

        train_dataset = DatasetWrapperForManualScoreClassificationWithMultiTask(
            train_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
            sigma=gaussian_sigma
        ).get_dataset(batch_size=batch_size)

        test_dataset = DatasetWrapperForManualScoreClassificationWithMultiTask(
            test_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
            sigma=gaussian_sigma
        ).get_dataset(batch_size=batch_size)

        # Create Input layer
        inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

        augmentation: bool = data_augmentation is not None

        model = create_cnn_manual_score_classification_multi_task_model_v1(
            inputs, 
            src.model.cnn.create_efficientNet_b4_pretrained, 
            augmentation=augmentation,
            zoom_range=data_augmentation['zoom_range'],
            rotation_range=data_augmentation['rotation_range'],
            num_classes=8,
            trainable=False,
            pooling=True # add 'avg' pooling for MLP head
        )

        # Save model structure for debugging 
        model_json = model.to_json(indent=2)
        model_json_path = os.path.join(fold_project_path, "model_structure.json")
        with open(model_json_path, "w", encoding='utf-8') as fs:
            fs.write(model_json)

        csv_log_path = os.path.join(fold_project_path, "training_log.csv")

        # Create checkpoint directory
        checkpoint_path = os.path.join(fold_project_path, "checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        callbacks = [
            # Set EarlyStopping. 
            # Monitor loss. stop training when no improvement in 5 epochs.
            EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1), 

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
            # Use KLDivergence for softmax and soft-label
            loss={
                "manual_score": tf.keras.losses.KLDivergence(), 
                "color_lighting_score": tf.keras.losses.KLDivergence(),
                "costume_detail_score": tf.keras.losses.KLDivergence(), 
                "proportion_score": tf.keras.losses.KLDivergence()
            }, 
            loss_weights={
                "manual_score": 1.0, 
                "color_lighting_score": 0.5,
                "costume_detail_score": 0.5,
                "proportion_score": 0.5
            }, 
            metrics={
                "manual_score": [expected_mae],
                "color_lighting_score": [expected_mae],
                "costume_detail_score": [expected_mae],
                "proportion_score": [expected_mae]
            }
        )

        model_history = model.fit(
            train_dataset, 
            validation_data=test_dataset,
            epochs=epoch_count,
            callbacks=callbacks,
            verbose=1
        )

        model_history_path = os.path.join(fold_project_path, "model_history.json")
        save_history_as_json(model_history, model_history_path)

        # Save model as keras each fold
        model_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}.keras")
        model.save(model_path)

        model_artifact_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}-TF-SavedModel")

        # Save model as TF SavedModel
        # See following link. 
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
        model.export(model_artifact_path)

        project_setting['project_path'] = fold_project_path

        fine_tuning_pre_trained_based_model(
            project_setting=project_setting,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            # Use KLDivergence for softmax and soft-label
            loss={
                "manual_score": tf.keras.losses.KLDivergence(), 
                "color_lighting_score": tf.keras.losses.KLDivergence(),
                "costume_detail_score": tf.keras.losses.KLDivergence(), 
                "proportion_score": tf.keras.losses.KLDivergence()
            }, 
            loss_weights={
                "manual_score": 1.0, 
                "color_lighting_score": 0.5,
                "costume_detail_score": 0.5,
                "proportion_score": 0.7
            }, 
            metrics={
                "manual_score": [expected_mae],
                "color_lighting_score": [expected_mae],
                "costume_detail_score": [expected_mae],
                "proportion_score": [expected_mae]
            },
            epoch=20,
            fine_tune_suffix=f"fine_tune_1_fold{fold + 1}",
            # unfreeze_boundary_name="block6a",
            unfreeze_boundary_name="block7a", # reduce unfreeze block count
            # new_learning_rate=1e-5
            new_learning_rate=5e-6
        )

def train_pseudo_label_generator_by_manual_score_regression_multi_task_k_fold(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict, 
):
    """
    Train CNN + MLP based pseudo-lable generator from manual labeled score. 

    Predict multi-task with `color_lighting_score` and `costume_detail_score`, `proportion_score`
    
    Using K-Fold Cross Validation
    """

    logger.info(f"Train {project_suffix} start...")

    # extract setting to dictionary
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

    # Save config file
    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load score records
    records = load_score_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Generate K-Fold index array
    # This will generate 
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    logger.info("Start K-Fold iteration")

    # K-Fold iteration
    for fold, (train_i, test_i) in enumerate(kfold.split(records)):
        logger.info(f"Start Fold {fold + 1}==========================")

        # Create sub directory each fold
        fold_project_path = os.path.join(project_path, f"fold_{fold + 1}")
        if not os.path.exists(fold_project_path):
            os.makedirs(fold_project_path)

        # Split train and test by K-Fold index array
        train_records = records.iloc[train_i]
        test_records = records.iloc[test_i]

        # Logging for debug
        logging_records(test_records, 'image_path', 'manual_score', 'color_lighting_score', 'costume_detail_score')

        # Export test records for debug and statistics
        test_records.to_csv(os.path.join(fold_project_path, "test_records.csv"))

        train_dataset = DatasetWrapperForManualScoreRegressionWithMultiTask(
            train_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
        ).get_dataset(batch_size=batch_size)

        test_dataset = DatasetWrapperForManualScoreRegressionWithMultiTask(
            test_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNetB7
        ).get_dataset(batch_size=batch_size)

        # Create Input layer
        inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

        augmentation: bool = data_augmentation is not None

        model = create_cnn_manual_score_regression_multi_task_model_v1(
            inputs, 
            src.model.cnn.create_efficientNet_b1_pretrained, 
            augmentation=augmentation,
            zoom_range=data_augmentation['zoom_range'],
            rotation_range=data_augmentation['rotation_range'],
            trainable=False,
            pooling=True # add 'avg' pooling for MLP head
        )

        # Save model structure for debugging 
        model_json = model.to_json(indent=2)
        model_json_path = os.path.join(fold_project_path, "model_structure.json")
        with open(model_json_path, "w", encoding='utf-8') as fs:
            fs.write(model_json)

        csv_log_path = os.path.join(fold_project_path, "training_log.csv")

        # Create checkpoint directory
        checkpoint_path = os.path.join(fold_project_path, "checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        callbacks = [
            # Set EarlyStopping. 
            # Monitor loss. stop training when no improvement in 5 epochs.
            EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1), 

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

            # Use MSE for scalar regression outputs
            loss={
                "manual_score": tf.keras.losses.MeanSquaredError(),
                "color_lighting_score": tf.keras.losses.MeanSquaredError(),
                "costume_detail_score": tf.keras.losses.MeanSquaredError(),
                "proportion_score": tf.keras.losses.MeanSquaredError()
            },

            # You can still apply different weights per task
            loss_weights={
                "manual_score": 1.0,
                "color_lighting_score": 0.5,
                "costume_detail_score": 0.5,
                "proportion_score": 0.6
            },

            # Use MAE (or MSE) as additional metrics
            metrics={
                "manual_score": [tf.keras.metrics.MeanAbsoluteError()],
                "color_lighting_score": [tf.keras.metrics.MeanAbsoluteError()],
                "costume_detail_score": [tf.keras.metrics.MeanAbsoluteError()],
                "proportion_score": [tf.keras.metrics.MeanAbsoluteError()]
            }
        )

        model_history = model.fit(
            train_dataset, 
            validation_data=test_dataset,
            epochs=epoch_count,
            callbacks=callbacks,
            verbose=1
        )

        model_history_path = os.path.join(fold_project_path, "model_history.json")
        save_history_as_json(model_history, model_history_path)

        # Save model as keras each fold
        model_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}.keras")
        model.save(model_path)

        model_artifact_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}-TF-SavedModel")

        # Save model as TF SavedModel
        # See following link. 
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
        model.export(model_artifact_path)

        project_setting['project_path'] = fold_project_path

        fine_tuning_pre_trained_based_model(
            project_setting=project_setting,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            # Use MSE for scalar regression outputs
            loss={
                "manual_score": tf.keras.losses.MeanSquaredError(),
                "color_lighting_score": tf.keras.losses.MeanSquaredError(),
                "costume_detail_score": tf.keras.losses.MeanSquaredError(),
                "proportion_score": tf.keras.losses.MeanSquaredError()
            },
            loss_weights={
                "manual_score": 1.0,
                "color_lighting_score": 0.5,
                "costume_detail_score": 0.5,
                "proportion_score": 0.5
            },
            metrics={
                "manual_score": [tf.keras.metrics.MeanAbsoluteError()],
                "color_lighting_score": [tf.keras.metrics.MeanAbsoluteError()],
                "costume_detail_score": [tf.keras.metrics.MeanAbsoluteError()],
                "proportion_score": [tf.keras.metrics.MeanAbsoluteError()]
            },
            epoch=20,
            fine_tune_suffix=f"fine_tune_1_fold{fold + 1}",
            # unfreeze_boundary_name="block6a",
            unfreeze_boundary_name="block7a", # reduce unfreeze block count
            # new_learning_rate=1e-5
            new_learning_rate=5e-6
        )

def train_pseudo_label_generator_by_quality_binary_k_fold(
    root_path: str, db_path: str, aliases_path: str, project_suffix:str, config_path: str, config_dict: dict
):
    """
    Train CNN + MLP based pseudo-lable generator from quality_binary 

    Using K-Fold Cross Validation
    """

    logger.info(f"Train {project_suffix} start...")

    # extract setting to dictionary
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

    # Save config file
    with open(project_setting['config_save_path'], "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    logger.info(f"Load records from {db_path}")

    # Load quality_binary records
    records = load_quality_binary_records(root_path, db_path)

    # Filter image file not exists
    records = filter_image_exists(records)

    # Generate K-Fold index array
    # This will generate 
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    logger.info("Start K-Fold iteration")

    # K-Fold iteration
    for fold, (train_i, test_i) in enumerate(kfold.split(records)):
        logger.info(f"Start Fold {fold + 1}==========================")

        # Create sub directory each fold
        fold_project_path = os.path.join(project_path, f"fold_{fold + 1}")
        if not os.path.exists(fold_project_path):
            os.makedirs(fold_project_path)

        # Split train and test by K-Fold index array
        train_records = records.iloc[train_i]
        test_records = records.iloc[test_i]

        # Logging for debug
        logging_records(test_records, 'image_path', 'quality_binary')

        # Export test records for debug and statistics
        test_records.to_csv(os.path.join(fold_project_path, "test_records.csv"))

        train_dataset = DatasetWrapperForAestheticBinaryClassification(
            train_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNet
        ).get_dataset(batch_size=batch_size)

        test_dataset = DatasetWrapperForAestheticBinaryClassification(
            test_records, width=width, height=height,
            normalize=False, # normalize false for pre-trained EfficientNet
        ).get_dataset(batch_size=batch_size)

        # Create Input layer
        inputs = Input(shape=(height, width, 3), dtype=tf.float32, name="image_input")

        augmentation: bool = data_augmentation is not None

        model = create_cnn_quality_binary_classification_model(
            inputs, 
            # src.model.cnn.create_efficientNet_b0_pretrained,
            src.model.cnn.create_efficientNet_b1_pretrained, # Change Backbone
            augmentation=augmentation,
            zoom_range=data_augmentation['zoom_range'],
            rotation_range=data_augmentation['rotation_range'],
            trainable=False,
            pooling=True # add 'avg' pooling for MLP head
        )

        # Save model structure for debugging 
        model_json = model.to_json(indent=2)
        model_json_path = os.path.join(fold_project_path, "model_structure.json")
        with open(model_json_path, "w", encoding='utf-8') as fs:
            fs.write(model_json)

        csv_log_path = os.path.join(fold_project_path, "training_log.csv")

        # Create checkpoint directory
        checkpoint_path = os.path.join(fold_project_path, "checkpoints")
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
            # Use binary crossentropy for binary classification
            loss={
                "quality_prediction": "binary_crossentropy"
            }, 
            metrics={
                "quality_prediction": [
                    tf.keras.metrics.BinaryAccuracy(name="q_acc"),
                    tf.keras.metrics.Precision(name="q_precision"),
                    tf.keras.metrics.Recall(name="q_recall")
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

        model_history_path = os.path.join(fold_project_path, "model_history.json")
        save_history_as_json(model_history, model_history_path)

        # Save model as keras each fold
        model_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}.keras")
        model.save(model_path)

        model_artifact_path = os.path.join(fold_project_path, f"model-mixed_precision_fold{fold + 1}-TF-SavedModel")

        # Save model as TF SavedModel
        # See following link. 
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model?_gl=1*145jd63*_up*MQ..*_ga*MzAxMzM3NDQyLjE3NDY5ODYwNjg.*_ga_W0YLR4190T*czE3NDY5ODYwNjckbzEkZzAkdDE3NDY5ODY0MjckajAkbDAkaDA.
        model.export(model_artifact_path)

        project_setting['project_path'] = fold_project_path

        fine_tuning_pre_trained_based_model(
            project_setting=project_setting,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            # Use binary crossentropy for binary classification
            loss={
                "quality_prediction": "binary_crossentropy"
            }, 
            metrics={
                "quality_prediction": [
                    tf.keras.metrics.BinaryAccuracy(name="q_acc"),
                    tf.keras.metrics.Precision(name="q_precision"),
                    tf.keras.metrics.Recall(name="q_recall")
                ]
            },
            epoch=20,
            fine_tune_suffix=f"fine_tune_1_fold{fold + 1}",
            # unfreeze_boundary_name="block6a",
            unfreeze_boundary_name="block7a", # reduce unfreeze block count
            # unfreeze_boundary_name="block_13", # For MobileNetV2
            new_learning_rate=1e-5
            # new_learning_rate=5e-6
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

def expected_mae_for_sanity(y_true, y_pred):
    """
    Compute MAE from expected score of softmax output

    For sanity_level prediction

    class_indices = [2, 4, 6, 8]
    """
    # y_pred is softmax, y_true is soft label
  	# Replace NaNs to 0 for safety 
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred, dtype=tf.float32), y_pred)

    # Get Class index tensor
    class_indices = tf.cast([2, 4, 6, 8], tf.float32)

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

    # Override for serialize
    def get_config(self):
        config = {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def fine_tuning_pre_trained_based_model_with_warmup(
        project_setting, model: Model, 
        train_dataset, test_dataset, 
        loss, metrics, epoch, 
        total_steps, warmup_steps,
        steps_per_epoch = None,
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

    # Re calculate warm up step
    total_steps = (epoch * steps_per_epoch) if steps_per_epoch is not None else total_steps
    warmup_steps = int(0.1 * total_steps)

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

def execution_example_v5_1():
    """
    Function for experiment '5th_1_experiment_pre-trained_efficientnetb0_based_rating_classification'
    
    Change Backbone to EfficientNetB0
    """

    train_rating_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_r-18_sampling.sqlite3", "",
        "5th_1_experiment_pre-trained_efficientnetb0_based_rating_classification", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.15
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

def execution_example_v6_2():
    """
    Function for experiment '6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification`
    """

    train_ai_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag_uniform_sanity.sqlite3", 
        "", "6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.1
            }
        }
    )

def continuous_example_v6_2():
    """
    Fucntion for continuous learning of '6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification` from `fine_tune_1`
    """

    logger.info(f"continous training `6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification` from `fine_tune_1`")

    # Suffix
    continuous_suffix = "continuous_from_fine_tune_1"

    # Load with `compile=True` (Default)
    model = tf.keras.models.load_model("/data/PixivDataBookmarks/model_project_6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification/model-mixed_precision_fine_tune_1.keras")

    # Same as example_v6_2
    project_setting = get_project_setting(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag_uniform_sanity.sqlite3", 
        "", "6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.1
            }
        }
    )

    # Reload dataset
    # Same step as example_v6_2

    logger.info(f"Load records from {project_setting['db_path']}")

    records = load_records("/data/PixivDataBookmarks", project_setting['db_path'])

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'ai_prediction')

    width = project_setting['width']
    height = project_setting['height']

    # Set batch size for fine_tune
    batch_size = project_setting['batch_size']

    # Get dataset as example_v6_2
    train_dataset = DatasetWrapperForAiClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForAiClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    csv_log_path = os.path.join(project_setting['project_path'], f"training_log_{continuous_suffix}.csv")

    checkpoint_path = os.path.join(project_setting['project_path'], f"checkpoint_{continuous_suffix}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set callback for continuous training
    callbacks = [
        # Set EarlyStopping. 
        # Monitor loss. stop training when no improvement in 3 epochs.
        EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1), 

        # Save model for each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_path, "{epoch:02d}-{val_loss:.2f}.keras"),
                        monitor="val_loss", mode="min", save_weights_only=False, verbose=1),
        
        CSVLogger(csv_log_path),
		
		# Terminate when loss or metric is NaN for safety
		TerminateOnNaN()
    ]

    # Continue train 
    # only epoch 5
    model_history = model.fit(
        train_dataset, 
        validation_data=test_dataset,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    model_path = os.path.join(project_setting['project_path'], f"model-mixed_precision_{continuous_suffix}.keras")

    model.save(model_path)

    model_artifact_path = os.path.join(project_setting['project_path'], f"model-mixed_precision-TF-SavedModel_{continuous_suffix}")

    model.export(model_artifact_path)

def fine_tuning_2_example_v6_2():
    """
    For Fine-tuning experiment '6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification`

    Change unfreeze boundary from `block6a` to `bloack7a` 
    for reduce trainable parameter (because, prev fine-tuning was overfitted)
    """


    logger.info(f"fine-tuning `6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification` by unfreezing `bloack7a`")
    # Suffix
    fine_tune_suffix = "fine_tune_2"

    # Load with `compile=True` (Default)
    # Load backbone freezed model
    model = tf.keras.models.load_model("/data/PixivDataBookmarks/model_project_6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification/model-mixed_precision.keras")

    # Same as example_v6_2
    project_setting = get_project_setting(
        "/data/PixivDataBookmarks", ".database/metadata_base_sample_ai_flag_uniform_sanity.sqlite3", 
        "", "6-2th_experiment_pre-trained_efficientnetb4_mlp_ai_classification", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 30, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.1
            }
        }
    )

    # Reload dataset
    # Same step as example_v6_2

    logger.info(f"Load records from {project_setting['db_path']}")

    records = load_records("/data/PixivDataBookmarks", project_setting['db_path'])

    # Filter image file not exists
    records = filter_image_exists(records)

    # Split train and test
    train_records, test_records = train_test_split(
        records, test_size=0.3, random_state=42
    )

    # Logging for debug
    logging_records(test_records, 'image_path', 'ai_prediction')

    width = project_setting['width']
    height = project_setting['height']

    # Set batch size for fine_tune
    batch_size = project_setting['batch_size']

    # Get dataset as example_v6_2
    train_dataset = DatasetWrapperForAiClassification(
        train_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    test_dataset = DatasetWrapperForAiClassification(
        test_records, width=width, height=height,
        normalize=False # normalize false for pre-trained EfficientNetB4
    ).get_dataset(batch_size=batch_size)

    # Run fine tuning
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
        epoch=15,
        fine_tune_suffix=fine_tune_suffix,
        unfreeze_boundary_name="block7a", # unfreeze after bloack7a_expand_conv
        new_learning_rate=1e-5
    )

def execution_1st_pseudo_label_generator_train():
    """
    Function to train `1st_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    """

    train_pseudo_label_generator_by_manual_score(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "1st_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_2nd_pseudo_label_generator_train():
    """
    Function to train `2nd_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v2`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_2nd_1_pseudo_label_generator_train():
    """
    Function to train `2nd_1_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_manual_score_classification_model_v2`

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_1_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 0.001, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_2nd_2_pseudo_label_generator_train():
    """
    Function to train `2nd_2_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_manual_score_classification_model_v2`

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_2_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 0.001, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }, gaussian_sigma=0.7 # reduce gaussian sigma
    )

def execution_2nd_3_pseudo_label_generator_train():
    """
    Function to train `2nd_3_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_manual_score_classification_model_v2`

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_3_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 1e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }, gaussian_sigma=0.7 # reduce gaussian sigma
    )

def execution_2nd_4_pseudo_label_generator_train():
    """
    Function to train `2nd_4_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_manual_score_classification_model_v4`

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_4_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 1e-3, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.1
            }
        }
    )

def execution_2nd_5_pseudo_label_generator_train():
    """
    Function to train `2nd_5_pseudo-label_generator_pre-trained_efficientNetb0_mlp`
    
    use `create_cnn_manual_score_classification_model_v4`

    EfficientNetB0 based. Reduce augmentation

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_5_pseudo-label_generator_pre-trained_efficientNetb0_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 1e-3, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }
    )

def execution_2nd_6_pseudo_label_generator_train():
    """
    Function to train `2nd_6_pseudo-label_generator_pre-trained_mobilenetv2_mlp`
    
    use `create_cnn_manual_score_classification_model`

    MobileNetV2 based. Reduce augmentation

    Backbone change
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "2nd_6_pseudo-label_generator_pre-trained_mobilenetv2_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 1e-3, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }
    )

def execution_3rd_pseudo_label_generator_train():
    """
    Function to train `3rd_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v3`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "3rd_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }, gaussian_sigma=0.8 #reduce gaussian soft-label sigma
    )

def execution_4th_pseudo_label_generator_train():
    """
    Function to train `4th_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v4`

    change `unfreeze_boundary_name` from `block6a` to `block7a`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "4th_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.1, "rotation_range": 0.15
            }
        }, gaussian_sigma=0.8 #reduce gaussian soft-label sigma
    )

def execution_5th_pseudo_label_generator_train():
    """
    Function to train `5th_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v2`

    change `unfreeze_boundary_name` from `block6a` to `block7a`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "5th_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_6th_pseudo_label_generator_train():
    """
    Function to train `6th_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v5`

    change `unfreeze_boundary_name` from `block6a` to `block7a`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "6th_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_7th_pseudo_label_generator_train():
    """
    Function to train `7th_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_model_v6`

    change `unfreeze_boundary_name` from `block6a` to `block7a`
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "7th_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 1e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }, gaussian_sigma=0.7
    )

def execution_8th_pseudo_label_generator_train():
    """
    Function to train `8th_pseudo-label_generator_pre-trained_efficientNetb4_mlp`
    
    use `create_cnn_manual_score_classification_multi_task_model_v1`
    """

    train_pseudo_label_generator_by_manual_score_multi_task_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "8th_pseudo-label_generator_pre-trained_efficientNetb4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 1e-3, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }, gaussian_sigma=0.7
    )

def execution_9th_pseudo_label_generator_train():
    """
    Function to train `9th_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_manual_score_regression_multi_task_model_v1`
    """

    train_pseudo_label_generator_by_manual_score_regression_multi_task_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "9th_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 1e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.2
            }
        }
    )

def execution_10th_pseudo_label_generator_train():
    """
    Function to train `10th_pseudo-label_generator_pre-trained_efficientNetb0_mlp`
    
    use `create_cnn_manual_score_regression_multi_task_model_v4`

    With coarse 3-class soft-label 
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "10th_pseudo-label_generator_pre-trained_efficientNetb0_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 1e-3, 
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }, gaussian_sigma=0.7
    )
    
def execution_10th_1_pseudo_label_generator_train():
    """
    Function to train `10th_1_pseudo-label_generator_pre-trained_efficientNetb0_mlp`
    
    use `create_cnn_manual_score_regression_multi_task_model`

    `ffn_dim` = 256, `dropout_rate` = 0.3

    With coarse 3-class soft-label 
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "10th_1_pseudo-label_generator_pre-trained_efficientNetb0_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 1e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }, gaussian_sigma=0.7
    )

def execution_10th_2_pseudo_label_generator_train():
    """
    Function to train `10th_2_pseudo-label_generator_pre-trained_mobilenetv2_mlp`
    
    use `create_cnn_manual_score_regression_multi_task_model`

    `ffn_dim` = 256, `dropout_rate` = 0.3

    With coarse 3-class soft-label 
    """

    train_pseudo_label_generator_by_manual_score_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_manual_verified.sqlite3", "",
        "10th_2_pseudo-label_generator_pre-trained_mobilenetv2_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 1e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }, gaussian_sigma=0.7
    )

def execution_1st_sanity_level_soft_label_train():
    """
    Function for experiment `1st_sanity_level_soft_label_pre-trained_efficientNetB4_mlp`

    EfficientNetB4 Backbone

    Gaussian soft-label based sanity_level prediction
    """

    train_sanity_level_classification_model(
        "/data/PixivDataBookmarks", ".database/metadata_sanity_r-18_sampling.sqlite3", "",
        "1st_sanity_level_soft_label_pre-trained_efficientNetB4_mlp", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 0.001,
            'batch_size': 64, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.15
            }
        }, gaussian_sigma=0.7
    )

def execution_2nd_sanity_level_soft_label_train():
    """
    Function for experiment `2nd_sanity_level_soft_label_pre-trained_efficientNetB4_transformer`

    EfficientNetB4 Backbone

    Gaussian soft-label based sanity_level prediction
    """

    train_sanity_level_classification_transformer_model(
        "/data/PixivDataBookmarks", ".database/metadata_sanity_r-18_sampling.sqlite3", "",
        "2nd_sanity_level_soft_label_pre-trained_efficientNetB4_transformer", None, {
            'image_width': 380, 'image_height': 380, 'learning_rate': 1e-4,
            'batch_size': 32, 'epoch': 40, 'data_augmentation': {
                "zoom_range": 0.15, "rotation_range": 0.15
            }
        }, gaussian_sigma=0.7
    )

def execution_12th_pseudo_label_generator_train():
    """
    Function to train `12th_pseudo-label_generator_pre-trained_efficientNetb0_mlp`
    
    use `create_cnn_quality_binary_classification_model`

    Binary classification by `quality_binary` 
    """

    train_pseudo_label_generator_by_quality_binary_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_for_quality_binary.sqlite3", "",
        "12th_pseudo-label_generator_pre-trained_efficientNetb0_mlp", None, {
            'image_width': 224, 'image_height': 224, 'learning_rate': 5e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
            }
        }
    )

def execution_12th_1_pseudo_label_generator_train():
    """
    Function to train `12th_1_pseudo-label_generator_pre-trained_efficientNetb1_mlp`
    
    use `create_cnn_quality_binary_classification_model`

    Binary classification by `quality_binary` 

    Change Backbone to EfficientNetB1
    """

    train_pseudo_label_generator_by_quality_binary_k_fold(
        "/data/PixivDataBookmarks", ".database/metadata_for_quality_binary.sqlite3", "",
        "12th_1_pseudo-label_generator_pre-trained_efficientNetb1_mlp", None, {
            'image_width': 240, 'image_height': 240, 'learning_rate': 5e-4, 
            'batch_size': 32, 'epoch': 35, 'data_augmentation': {
                "zoom_range": 0.05, "rotation_range": 0.02
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
    # execution_example_v6_1()
    # execution_example_v6_2()
    # continuous_example_v6_2()
    # fine_tuning_2_example_v6_2()
    # execution_1st_pseudo_label_generator_train()
    # execution_2nd_pseudo_label_generator_train()
    # execution_3rd_pseudo_label_generator_train()
    # execution_4th_pseudo_label_generator_train()
    # execution_5th_pseudo_label_generator_train()
    # execution_6th_pseudo_label_generator_train()
    # execution_7th_pseudo_label_generator_train()
    # execution_8th_pseudo_label_generator_train()
    # execution_2nd_1_pseudo_label_generator_train()
    # execution_2nd_2_pseudo_label_generator_train()
    # execution_2nd_3_pseudo_label_generator_train()
    # execution_9th_pseudo_label_generator_train()
    # execution_2nd_4_pseudo_label_generator_train()
    # execution_2nd_5_pseudo_label_generator_train()
    # execution_10th_pseudo_label_generator_train()
    # execution_10th_1_pseudo_label_generator_train()
    # execution_2nd_6_pseudo_label_generator_train()
    # execution_10th_2_pseudo_label_generator_train()
    # execution_example_v5_1()
    # execution_1st_sanity_level_soft_label_train()
    # execution_2nd_sanity_level_soft_label_train()
    # execution_12th_pseudo_label_generator_train()
    execution_12th_1_pseudo_label_generator_train()