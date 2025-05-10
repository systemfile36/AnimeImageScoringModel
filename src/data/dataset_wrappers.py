import numpy as np
import tensorflow as tf

class DatasetWrapper():
    """
    DatasetWrapper class for data pipelining

    The dataset will have the shape `(image, { "score_prediction": ... })`

    This class generates a `tf.data.Dataset` for 
    the default score-prediction model.

    You can customize the dataset labels by overriding the following methods :

    `load_image_for_map`, `map_labels`. 
    (`map_labels` should call `map_labels_py` by using `tf.py_function`)

    The output shape of 'map_labels' should match the expected dataset structure.

    Mapping functions will be called in the following order : 
    'load_image_for_map', then 'map_labels'
    """

    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True
    ):
        """
        The argument 'data' must contain the keys 'image_path' and 'score_prediction'
        """

        # Create dictionary from argument
        self.inputs: dict[str, list] = { 
            'image_path': data['image_path'],  
            'score_prediction': data['score_prediction']
        } if data is not None else None

        self.width = width
        self.height = height

        self.normalize = normalize

    def get_dataset(self, batch_size: int) -> tf.data.Dataset:
        """
        Create `tf.data.Dataset` from `self.inputs`.

        Args:
            batch_size (int): minibatch size of `Dataset`

        Returns:
            tf.data.Dataset: `tf.data.Dataset` instance representing dataset. 
        """

        # slice dictionary to tensors
        ds = tf.data.Dataset.from_tensor_slices(self.inputs)

        # Load images from image_paths
        ds = ds.map(self.load_image_for_map, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.ignore_errors()

        ds = ds.map(self.map_labels, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

    def load_image_for_map(self, data_slice):
        """
        Loads image from `image_path`. 

        Override this method to fit the shape of your dataset.
        """

        image = self.load_image(data_slice['image_path'])

        # unpack data_slice and return only necessary 
        return (image, {
            'score_prediction': data_slice['score_prediction']
        })

    @tf.function
    def load_image(self, image_path: str):
        """
        Load an image with from `image_path`, resized it with padding.

        This method uses only Tensorflow functions.

        Args:
            image_path (str): Path to the image to be loaded.
        
        Returns:

        """
        image_raw = tf.io.read_file(image_path)

        image = tf.io.decode_png(image_raw, channels=3)

        # Resize to model Input size with padding (to preserve aspect ratio and fix shape)
        image = tf.image.resize_with_pad(
            image, 
            target_height=self.height,
            target_width=self.width,
            method=tf.image.ResizeMethod.AREA
        )

        # normalize to [0, 1) by convert to tf.float32
        if self.normalize:
            image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def map_labels(self, image, data_slice):
        return (image, data_slice)

    def map_labels_py(self, data_slice):
        pass

class DatasetWithMetaWrapper(DatasetWrapper):
    """
    DatasetWrapper class for dataset with metadata. 

    The dataset will have the shape 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one hot vector", 
        'score_prediction': "float32"
        })
    ```
        
    The input data shuold have the shape 
    ({
        'image_path': "str, absolute path to image",
        'ai_prediction': "int, 0 or 1",
        'rating_prediction': "int, 2 or 4 or 6"
        'score_prediction': "int or float32"
    })

    """
    def __init__(self, data: dict, width: int, height: int, normalize: bool=True):
        
        # Call parents `__init__` to initialize member fields
        super().__init__(None, width, height, normalize)

        self.inputs = {
            'image_path': data['image_path'],
            'ai_prediction': data['ai_prediction'],
            'rating_prediction': data['rating_prediction'],
            'score_prediction': data['score_prediction']
        }


    def load_image_for_map(self, data_slice):

        image = self.load_image(data_slice['image_path'])

        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['ai_prediction'],
                data_slice['rating_prediction'],
                data_slice['score_prediction'])
    
    def map_labels(self, image, ai_value, rating_value, score_value):
        image, ai, rating, score = tf.py_function(
            self.map_labels_py,
            (image, ai_value, rating_value, score_value),
            (tf.float32, tf.float32, tf.float32, tf.float32)
        )

        # Set shape for safety
        ai.set_shape([1]) #0.0 or 1.0
        rating.set_shape([3]) # (3,) one-hot vector
        score.set_shape([1]) # just float32
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        # packing to dict again
        return (image, {
            'ai_prediction': ai,
            'rating_prediction': rating,
            'score_prediction': score
        })

    def map_labels_py(self, image, ai_value, rating_value, score_value):
        
        # Value list of rating 
        rating_values = np.array([2, 4, 6])

        # for each items in rating_values, if value is equal to argument, then 1.0, otherwise, 0.0
        rating_one_hot = np.where(
            rating_values == rating_value, 1.0, 0.0
            ).astype(np.float32)
        
        # convert to np.array(..., dtype=np.float32) 
        # Because the function that execute by `tf.py_function` should return numpy or tf.Tensor
        ai_flag = np.array([ai_value]).astype(np.float32)

        # convert to np.float32 
        score = np.array([score_value]).astype(np.float32)

        # tf.py_function is not allow return dict. so return tuple
        return (image, ai_flag, rating_one_hot, score)

        
class DatasetWithMetaAndTagCharacterWrapper(DatasetWrapper):
    """
    DatasetWrapper class for dataset with metadata and tag_character. 

    The dataset will have the shape 

    ```
    (image, {
        'ai_prediction': "(1,), float32, 0 or 1", 
        'rating_prediction': "(3,), float32, one-hot vector", 
        'score_prediction': "float32",
        'tag_prediction': "(N,), multi-hot vector , N is `len(tag_character_all)`
        })
    ```

    The input data shuold have the shape 

    ```
    ({
        'image_path': "str, absolute path to image",
        'ai_prediction': "int, 0 or 1",
        'rating_prediction': "int, 2 or 4 or 6"
        'score_prediction': "int or float32"
        'tag_prediction': "comma-seperated strings"
    })
    ```

    """

    def __init__(self, data: dict, width: int, height: int, tag_character_all: list[str], normalize: bool=True):
     
        # Call parents `__init__` to initialize member fields
        super().__init__(None, width, height, normalize)

        self.inputs = {
            'image_path': data['image_path'],
            'ai_prediction': data['ai_prediction'],
            'rating_prediction': data['rating_prediction'],
            'score_prediction': data['score_prediction'],
            'tag_prediction': data['tag_prediction']
        }

        self.tag_character_all = tag_character_all
    
    def load_image_for_map(self, data_slice):

        image = self.load_image(data_slice['image_path'])

        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['ai_prediction'],
                data_slice['rating_prediction'],
                data_slice['score_prediction'], 
                data_slice['tag_prediction'])
    
    def map_labels(self, image, ai_value, rating_value, score_value, tag_string):
        image, ai, rating, score, tag = tf.py_function(
            self.map_labels_py, 
            (image, ai_value, rating_value, score_value, tag_string),
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        # Set shape for safety
        ai.set_shape([1]) #0.0 or 1.0
        rating.set_shape([3]) # (3,) one-hot vector
        score.set_shape([1]) # just float32
        tag.set_shape([len(self.tag_character_all)]) # (N,) multi-hot vector
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        # packing to dict again
        return (image, {
            'ai_prediction': ai,
            'rating_prediction': rating,
            'score_prediction': score,
            'tag_prediction': tag
        })


    def map_labels_py(self, image, ai_value, rating_value, score_value, tag_string):
        
        # Value list of rating 
        rating_values = np.array([2, 4, 6])

        # for each items in rating_values, if value is equal to argument, then 1.0, otherwise, 0.0
        rating_one_hot = np.where(
            rating_values == rating_value, 1.0, 0.0
            ).astype(np.float32)
        
        # convert to np.array(..., dtype=np.float32) 
        # Because the function that execute by `tf.py_function` should return numpy or tf.Tensor
        ai_flag = np.array([ai_value]).astype(np.float32)

        # convert to np.float32 
        score = np.array([score_value]).astype(np.float32)

        # Get 'tag_string' tensor by numpy array and decode to string
        tag_string: str = tag_string.numpy().decode()
        
        # Split to array
        tag_array = np.array(tag_string.split(","))

        # Convert to multi-hot vector. 
        tag_multi_hot = np.where(
            np.isin(self.tag_character_all, tag_array), 1.0, 0.0
        ).astype(np.float32)

        # tf.py_function is not allow return dict. so return tuple
        return (image, ai_flag, rating_one_hot, score, tag_multi_hot)