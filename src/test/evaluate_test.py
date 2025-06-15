import os
# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy.stats import pearsonr

from tensorflow.keras import Model

from src.data.dataset_wrappers import DatasetWrapperForAestheticBinaryClassification

from src.data.dataset import load_quality_binary_records

from src.data.dataset import load_score_records

@tf.function
def load_image(image_path: str, width: int, height: int):
        
        image_raw = tf.io.read_file(image_path)

        image = tf.io.decode_png(image_raw, channels=3)

        # Resize to model Input size with padding (to preserve aspect ratio and fix shape)
        image = tf.image.resize_with_pad(
            image, 
            target_height=height,
            target_width=width,
            method=tf.image.ResizeMethod.AREA
        )

        return image

def eval_quality_binary_model(model_project_path: str, model_path: str, dataset_root: str, db_path: str, 
                                width: int, height: int):
    
    # Load model for evaluate
    model = tf.keras.models.load_model(os.path.join(model_project_path, model_path), compile=True)

    # Load manual score for evaluation
    df = load_score_records(dataset_root, db_path)

    # Map `manual_score` to `quality_binary` by criteria
    df['quality_binary'] = df['manual_score'].apply(
        lambda x: 1 if x >= 8 else (0 if x < 5 else np.nan)
    )

    # filter na
    df = df[df['quality_binary'].notna()]

    eval_dataset = DatasetWrapperForAestheticBinaryClassification(
          df, width, height, normalize=False
    ).get_dataset(batch_size=32)

    result = model.evaluate(eval_dataset, verbose=2, return_dict=True)

    # Save result
    with open(os.path.join(model_project_path, "evalute_result.json"), "w", encoding="utf-8") as fs:
        json.dump(result, fs, indent=2)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate quality_binary model")

    parser.add_argument(
         "--model-project-path", type=str, 
         required=True,
         help="path of directory contain model"
    )

    parser.add_argument(
         "--model-path", type=str, required=True, help="relative path to model project path"
    )

    parser.add_argument(
         "--dataset-root", type=str, required=True, help="Dataset root"
    )

    parser.add_argument(
         "--db-path", type=str, required=True, help="relative path to 'manual_score' db"
    )

    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=('WIDTH', 'HEIGHT'),
        required=True,
        help="Target size of resize. ex) --size 512 512."
    )
    args = parser.parse_args()

    size = tuple(args.size)

    eval_quality_binary_model(args.model_project_path, args.model_path, args.dataset_root, args.db_path, width=size[0], height=size[1])