
import os
# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from tensorflow.keras import Model

from src.data.dataset_wrappers import DatasetWrapperForManualScoreClassification

WIDTH = 240
HEIGHT = 240

@tf.function
def load_image(image_path: str):
        
        image_raw = tf.io.read_file(image_path)

        image = tf.io.decode_png(image_raw, channels=3)

        # Resize to model Input size with padding (to preserve aspect ratio and fix shape)
        image = tf.image.resize_with_pad(
            image, 
            target_height=HEIGHT,
            target_width=WIDTH,
            method=tf.image.ResizeMethod.AREA
        )

        return image

if __name__ == "__main__":

    # project_path to process
    project_path = "/data/PixivDataBookmarks/model_project_2nd_4_pseudo-label_generator_pre-trained_efficientNetb1_mlp/fold_4"
    # Load model 
    model = tf.keras.models.load_model(
        os.path.join(project_path, "model-mixed_precision_fold4.keras"), compile=False)
    
    # Load validation records 
    # Use `test_records.csv` to avoid data leak
    df = pd.read_csv(os.path.join(project_path, "test_records.csv"))

    # Use `image_path` only to inference
    ds = tf.data.Dataset.from_tensor_slices(df['image_path'])

    # Load image from `image_path`
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.ignore_errors()

    ds = ds.batch(batch_size=32)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # shape: (batch, 8)
    y_pred = model.predict(ds)

    class_centors = np.linspace(3, 10, 8)

    # axis=1 because y_pred has batch dimension 
    y_pred_scores = np.sum(y_pred['manual_score'] * class_centors, axis=1)

    # ground truth
    y_true_scores = df['manual_score']

    # Print correlation coefficient
    pearson_corr, _ = pearsonr(y_true_scores, y_pred_scores)
    print(f"pearson correlation coefficient: {pearson_corr}")

    # Draw scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_scores, y_pred_scores, alpha=0.6)

    # Guide line
    plt.plot([3, 10], [3, 10], color='red', linestyle='--', label="y = x")
    
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs. True Score (Expected Value of Softmax Output)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("/data/correlation_scatter_2nd_4_pseudo-label_ganerator_fold_4.png")

    class_indices = class_centors.reshape(1, -1)  # shape: (1, 8)
    confidence = np.max(y_pred['manual_score'], axis=1)  # shape: (batch,)
    expected = y_pred_scores  # already computed
    variance = np.sum(y_pred['manual_score'] * (class_indices - expected[:, None]) ** 2, axis=1)

    # Boolean mask
    mask = (confidence > 0.2) & (variance < 2.5)

    # Filtered samples
    filtered_true = y_true_scores[mask]
    filtered_pred = y_pred_scores[mask]

    # Compute correlation coefficients
    pearson_filtered, _ = pearsonr(filtered_true, filtered_pred)
    print(f"filtered pearson correlation coefficient: {pearson_filtered}")

    # Draw filtered scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_scores, y_pred_scores, alpha=0.3, label="All Samples")
    plt.scatter(filtered_true, filtered_pred, alpha=0.8, color='green', label="Filtered Samples")
    plt.plot([3, 10], [3, 10], color='red', linestyle='--', label="y = x")

    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title(f"Filtered: Predicted vs. True Score (Conf > 0.2 & Var < 2.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("/data/correlation_scatter_2nd_4_filtered_conf_var_fold_4.png")

    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(confidence, variance, alpha=0.3)
    plt.xlabel("Confidence (max prob)")
    plt.ylabel("Predicted Variance")
    plt.title("Confidence vs. Variance Distribution")
    plt.grid(True)

    plt.savefig("/data/correlation_scatter_2nd_4_confidence_variance_fold_4.png")