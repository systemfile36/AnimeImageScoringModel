# Test for Dataset pipelining

from src.data import DatasetWrapper, DatasetWithMetaWrapper, DatasetWithMetaAndTagCharacterWrapper
import os

# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

#tf.debugging.set_log_device_placement(True)

import random
from PIL import Image
import shutil

# Size of test dataset
TEST_SIZE = 5000
TEST_DIR = "temp_images"

# Generate temp directory for test
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

def generate_random_image_tensor(width, height):
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

shapes = [
    (1024, 768),
    (800, 600),
    (1280, 800),
    (450, 450)
]

# Generate random image with random shape.
image_paths = []
for i, (w, h) in enumerate(shapes):
    image = generate_random_image_tensor(w, h)
    file_path = os.path.join(TEST_DIR, f"temp_{i}.png")
    Image.fromarray(image).save(file_path)
    image_paths.append(file_path)

# Generate test records.
test_path = [image_paths[i % len(image_paths)] for i in range(TEST_SIZE)]

test_score = [random.randrange(1, 100) for _ in range(TEST_SIZE)]

test_ai_flag = [random.choice([0, 1]) for _ in range(TEST_SIZE)]

test_rating = [random.choice([2, 4, 6]) for _ in range(TEST_SIZE)]

test_tag_all = [
    'hatsune_miku', 'hakurei_reimu', 'kirisame_marisa', 
    'remilia_scarlet', 'shiroko_(blue_archive)'
]

# Random sampling from test_tag
test_tag = [','.join(random.sample(test_tag_all, random.randrange(0, len(test_tag_all)))) 
                for _ in range(TEST_SIZE)]

def print_shape(name: str, dataset: tf.data.Dataset):
    tf.print(name)
    for image_batch, label_batch in dataset.take(1):
        tf.print("Image shape: ", image_batch.shape)
        tf.print(label_batch)

def wipe_temp():
    if os.path.exists(TEST_DIR) and os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)

if __name__ == "__main__":

    print(tf.config.list_physical_devices())

    # test data 
    dataset = DatasetWrapper(
        {
            'image_path': test_path,
            'score_prediction': test_score
        }, 
        512, 512
    )

    print_shape("DatasetWrapper", dataset.get_dataset(32))
    # print(dataset.get_dataset(32).take(1))


    dataset = DatasetWithMetaWrapper(
        {
            'image_path': test_path,
            'ai_prediction': test_ai_flag,
            'rating_prediction': test_rating,
            'score_prediction': test_score
        }, 
        512, 512
    )

    print_shape("DatasetWithMetaWrapper", dataset.get_dataset(32))
    # print(dataset.get_dataset(32).take(1))

    dataset = DatasetWithMetaAndTagCharacterWrapper(
        {
            'image_path': test_path,
            'ai_prediction': test_ai_flag,
            'rating_prediction': test_rating,
            'score_prediction': test_score,
            'tag_prediction': test_tag
        }, 
        512, 512, 
        tag_character_all=test_tag_all
    )

    print_shape("DatasetWithMetaAndTagCharacterWrapper", dataset.get_dataset(32))

    wipe_temp()