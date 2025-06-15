import os
# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import sqlite3
import argparse

from src.utils import get_filename_based_logger

from src.data.dataset_wrappers import DatasetWrapperForAestheticBinaryClassification

from src.data.dataset import load_records

logger = get_filename_based_logger(__file__)

# global size variable for `load_image`
width_global = 224
height_global = 224

@tf.function
def load_image(image_path: str):
        
        image_raw = tf.io.read_file(image_path)

        image = tf.io.decode_png(image_raw, channels=3)

        # Resize to model Input size with padding (to preserve aspect ratio and fix shape)
        image = tf.image.resize_with_pad(
            image, 
            target_height=height_global,
            target_width=width_global,
            method=tf.image.ResizeMethod.AREA
        )

        return image

def pseudo_labeling_data(dataset_root: str, db_path: str, pseudo_labeling_model_path: str,
                         confidence_positive_lower_bound: float = 0.9, confidence_negative_upper_bound: float = 0.1, 
                         width: int = 224, height: int = 224):
    """
    Pseudo-labeling for quality_binary classification

    filter by confidence. 
    
    If `p > confidence_positive_lower_bound` then `quality_binary` is 1
    
    If `p < confidence_negative_upper_bound` then `quality_binary` is 0

    Otherwise, no update (Default is NULL)
    """

    logger.info(f"Load Pseudo-labeling model {pseudo_labeling_model_path}")

    model = tf.keras.models.load_model(pseudo_labeling_model_path, compile=False)

    # global size variable for `load_image`
    global width_global, height_global
    width_global = width
    height_global = height

    logger.info(f"Load records from {db_path}")

    df = load_records(dataset_root, db_path)

    # Extract `image_path` only for inference
    image_paths = df['image_path'].to_list()

    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.ignore_errors()

    ds = ds.batch(batch_size=64)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    logger.info("Start predict...")

    # shape: (batch, 1), sigmoid binary vector
    y_pred = model.predict(ds)

    # Extract dictionary
    y_pred = y_pred['quality_prediction']

    # (batch, 1) -> (batch, )
    y_pred = np.squeeze(y_pred)

    logger.info(f"Open {db_path} for update column")

    if not os.path.isabs(db_path):
         db_path = os.path.join(dataset_root, db_path)

    # Open DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.info("Clear quality_binary column")
    # Clear quality_binary column
    cursor.execute("""
    UPDATE illusts SET quality_binary = NULL;
    """)

    updated_count = 0

    for i, image_path in enumerate(image_paths):

        # Get sigmoid value 
        prob = float(y_pred[i])

        # Filter by confidence
        if prob > confidence_positive_lower_bound:
            label = 1
        elif prob < confidence_negative_upper_bound:
            label = 0
        else:
            continue  # Skip uncertain predictions

        # Extract file name without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Update column
        cursor.execute("""
            UPDATE illusts
            SET quality_binary = ?
            WHERE filename = ?
        """, (label, filename))

        updated_count += 1

    conn.commit()
    conn.close()

    logger.info(f"{updated_count} rows updated")

def pseudo_labeling_with_kfold_average(dataset_root: str, db_path: str, model_paths: list,
                                       confidence_positive=0.9, confidence_negative=0.1,
                                       width=224, height=224):
    """
    Pseudo-labeling using K-Fold model ensemble.

    - Average predictions → `quality_soft`
    - Threshold-based hard label → `quality_level`
    """

    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.info("Invalid model paths")
            return

    global width_global, height_global
    width_global = width
    height_global = height

    logger.info(f"Load records from {db_path}")

    # Extract `image_path` only for inference
    df = load_records(dataset_root, db_path)
    image_paths = df['image_path'].tolist()

    # Create input dataset
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.batch(batch_size=64).prefetch(tf.data.AUTOTUNE)

    # Predict with all models and collect probabilities
    all_preds = []

    for path in model_paths:
        logger.info(f"Start predict with model : {path}")
        model = tf.keras.models.load_model(path, compile=False)
        pred_dict = model.predict(ds)
        pred = pred_dict['quality_prediction'] if isinstance(pred_dict, dict) else pred_dict
        all_preds.append(np.squeeze(pred))  # (N,)

    # Stack and average: (num_models, N) → (N,)
    all_preds = np.stack(all_preds, axis=0)  # (K, N)
    avg_preds = np.mean(all_preds, axis=0)   # (N,)

    if not os.path.isabs(db_path):
         db_path = os.path.join(dataset_root, db_path)

    # Connect to SQLite DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    updated = 0

    for i, image_path in enumerate(image_paths):
        prob = float(avg_preds[i])
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Update quality_soft 
        cursor.execute("""
            UPDATE illusts SET quality_soft = ? WHERE filename = ?
        """, (prob, filename))

        # quality_level threshold check
        if prob > confidence_positive:
            label = 1
        elif prob < confidence_negative:
            label = 0
        else:
            continue  # skip uncertain predictions

        # quality_level update
        cursor.execute("""
            UPDATE illusts SET quality_binary = ? WHERE filename = ?
        """, (label, filename))

        updated += 1

    conn.commit()
    conn.close()

    logger.info(f"{updated} rows updated")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run pseudo-labeling for binary quality classification.")

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset (for resolving image paths).")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to the SQLite database file.")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="List of paths to trained fold models (e.g. fold_0.keras fold_1.keras ...)")
    parser.add_argument("--conf_pos", type=float, default=0.9,
                        help="Confidence threshold above which quality_binary=1.")
    parser.add_argument("--conf_neg", type=float, default=0.1,
                        help="Confidence threshold below which quality_binary=0.")
    parser.add_argument("--width", type=int, default=224,
                        help="Target image width for model input.")
    parser.add_argument("--height", type=int, default=224,
                        help="Target image height for model input.")

    args = parser.parse_args()

    
    if len(args.model_paths) == 1:
        pseudo_labeling_data(
            dataset_root=args.dataset_root,
            db_path=args.db_path,
            pseudo_labeling_model_path=args.model_paths[0],
            confidence_positive_lower_bound=args.conf_pos,
            confidence_negative_upper_bound=args.conf_neg,
            width=args.width,
            height=args.height
        )
    else:
        pseudo_labeling_with_kfold_average(
            dataset_root=args.dataset_root,
            db_path=args.db_path,
            model_paths=args.model_paths,
            confidence_positive=args.conf_pos,
            confidence_negative=args.conf_neg,
            width=args.width,
            height=args.height
        )