# Test for Dataset pipelining

from src.data import DatasetWrapper, DatasetWithMetaWrapper
import os

# Supress warning. 
# Ignore all WARNING. only logging ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#tf.debugging.set_log_device_placement(True)

import random

# Size of test dataset
TEST_SIZE = 5000

# Generate random data 
test_path = ["/data/DataScripts/Temp/129282054_p0.png" for _ in range(TEST_SIZE)]

test_score = [random.randrange(1, 100) for _ in range(TEST_SIZE)]

test_ai_flag = [random.choice([0, 1]) for _ in range(TEST_SIZE)]

test_rating = [random.choice([2, 4, 6]) for _ in range(TEST_SIZE)]

def print_shape(name: str, dataset: tf.data.Dataset):
    tf.print(name)
    for image_batch, label_batch in dataset.take(1):
        tf.print("Image shape: ", image_batch.shape)
        tf.print(label_batch)

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
    

