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

        # Convert dtype of image to tf.float32 and normalize.
        if self.normalize:
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255.0

        return image

    def map_labels(self, image, data_slice):
        return (image, data_slice)

    def map_labels_py(self, data_slice):
        pass

class DatasetWrapperForAestheticBinaryClassification(DatasetWrapper):
    """
    DatasetWrapper class for image aesthetic binary classification

    This class used for 
    pseudo-label generator for image aesthetic binary classification 
    based on `quality_binary`

    The dataset will have the shape 

    ```
    (image, {
        'quality_prediction': "0 or 1, if hight quality then 1 else 0"
    })
    ```

    The input data should have the shape 

    ```
    ({
        'image_path': "str, absolute path to image", 
        'quality_binary': "0 or 1"
    })
    ```
    """

    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True
    ):
        """
        The argument `data` must contain the keys `image_path` and `quality_binary`
        """

        self.inputs = {
            'image_path': data['image_path'],
            'quality_binary': data['quality_binary']
        } if data is not None else None

        self.width = width
        self.height = height

        self.normalize = normalize

    def load_image_for_map(self, data_slice):
        
        image = self.load_image(data_slice['image_path'])
        
        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['quality_binary'])
    
    def map_labels(self, image, quality_binary):
        image, quality_vector = tf.py_function(
            self.map_labels_py,
            (image, quality_binary),
            (tf.float32, tf.float32)
        )

        # Set shape for safety
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image
        quality_vector.set_shape([1]) # binary vector

        # packing to dict again
        return (image, {
            'quality_prediction': quality_vector
        })

    def map_labels_py(self, image, quality_binary):
        
        # Just cast type to float32
        quality_vector = np.array([quality_binary]).astype(np.float32)

        return (image, quality_vector)


class DatasetWrapperForManualScoreClassification(DatasetWrapper):
    """
    DatasetWrapper class for `manual_score` classification

    This class used for `manual_score` prediction by soft-vector based classification 

    Use `num_classes` for coarse-lebel (Default is 8)

    The dataset will have the shape 

    ```
    (image, {
        'manual_score': "(N,), Soft label vector."
    })
    ```

    The input data should have the shape 

    ```
    ({
        `image_path`: "str, absolute path to image", 
        'manual_score': "float32, [3, 10]"
    })
    ```
    """

    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True, 
            sigma: float = 1.0, num_classes: int = 8
    ):
        """
        The argument `data` must contain the keys `image_path` and `manual_score`
        """

        self.inputs = {
            'image_path': data['image_path'],
            'manual_score': data['manual_score']
        } if data is not None else None

        self.width = width
        self.height = height

        self.normalize = normalize

        self.num_classes = num_classes

        # sigma for Gaussian soft-label
        self.sigma = sigma

    def load_image_for_map(self, data_slice):
        
        image = self.load_image(data_slice['image_path'])
        
        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['manual_score'])
    
    def map_labels(self, image, score_value):
        image, score_vector = tf.py_function(
            self.map_labels_py,
            (image, score_value),
            (tf.float32, tf.float32)
        )

        # Set shape for safety
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image
        score_vector.set_shape([self.num_classes]) # (N,), soft label vector

        # packing to dict again
        return (image, {
            'manual_score': score_vector
        })

    def map_labels_py(self, image, score_value):
        
        # Convert score to soft label vector 
        # (8,) like [0, 0, ..., 0.4, 0.6, ...]
        # score = self.score_to_soft_label(score_value)

        # Use Gaussian soft label instead of Linear
        score = self.score_to_gaussian_soft_label(score_value, self.sigma)

        return (image, score)
    
    def score_to_gaussian_soft_label(self, score, sigma: float=1.0) -> np.ndarray:
        """
        Convert a float score (3 ~ 10) to a Gaussian based soft label (length = num_classes)
        
        Args:
            score: float in range [3, 10]
            sigma: standard deviation of Gaussian kernel (controls smoothness)
        """

        # Class centers: 
        class_centers = np.linspace(3, 10, self.num_classes)

        # Compute Gaussian probabilities
        label = np.exp(- (score - class_centers) ** 2 / (2 * sigma ** 2))

        # Normalize
        label /= np.sum(label)

        return label.astype(np.float32)

class DatasetWrapperForManualScoreClassificationWithMultiTask(DatasetWrapper):
    """
    DatasetWrapper class for `manual_score` classification

    This class used for `manual_score` prediction by soft-vector based classification 

    The dataset will have the shape 

    ```
    (image, {
        'manual_score': "(8,), Soft label vector.",
        'color_lighting_score': "(8,), Soft label vector.",
        'costume_detail_score': "(8,), Soft label vector.", 
        'proportion_score': "(8,), Soft label vector."
    })
    ```

    The input data should have the shape 

    ```
    ({
        `image_path`: "str, absolute path to image", 
        'manual_score': "float32, [3, 10]",
        'color_lighting_score': "float32, [3, 10]",
        'costume_detail_score': "float32, [3, 10]",
        'proportion_score': "float32, [3, 10]"
    })
    ```
    """

    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True, 
            sigma: float = 1.0
    ):
        """
        The argument `data` must contain the keys `image_path` and `manual_score`
        """

        self.inputs = {
            'image_path': data['image_path'],
            'manual_score': data['manual_score'], 
            'color_lighting_score': data['color_lighting_score'],
            'costume_detail_score': data['costume_detail_score'], 
            'proportion_score': data['proportion_score']
        } if data is not None else None

        self.width = width
        self.height = height

        self.normalize = normalize

        # length of `manual_score` soft label vector
        self.num_classes = 8

        # sigma for Gaussian soft-label
        self.sigma = sigma

    def load_image_for_map(self, data_slice):
        
        image = self.load_image(data_slice['image_path'])
        
        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['manual_score'], 
                data_slice['color_lighting_score'],
                data_slice['costume_detail_score'],
                data_slice['proportion_score'])
    
    def map_labels(self, image, manual_score, color_lighting_score, costume_detail_score, proportion_score):
        image, manual, color_lighting, costume_detail, proportion = tf.py_function(
            self.map_labels_py,
            (image, manual_score, color_lighting_score, costume_detail_score, proportion_score),
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        # Set shape for safety
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image
        manual.set_shape([self.num_classes]) # (N,), soft label vector
        color_lighting.set_shape([self.num_classes]) # (N,), soft label vector
        costume_detail.set_shape([self.num_classes]) # (N,), soft label vector
        proportion.set_shape([self.num_classes]) # (N,), soft label vector

        # packing to dict again
        return (image, {
            'manual_score': manual, 
            'color_lighting_score': color_lighting, 
            'costume_detail_score': costume_detail,
            'proportion_score': proportion
        })

    def map_labels_py(self, image, manual_score, color_lighting_score, costume_detail_score, proportion_score):
        
        # Convert score to soft label vector 
        # (8,) like [0, 0, ..., 0.4, 0.6, ...]
        # score = self.score_to_soft_label(score_value)

        # Use Gaussian soft label instead of Linear
        manual = self.score_to_gaussian_soft_label(manual_score, self.sigma)

        color_lighting = self.score_to_gaussian_soft_label(color_lighting_score, self.sigma)

        costume_detail = self.score_to_gaussian_soft_label(costume_detail_score, self.sigma)

        proportion = self.score_to_gaussian_soft_label(proportion_score, self.sigma)

        return (image, manual, color_lighting, costume_detail, proportion)
    
    def score_to_gaussian_soft_label(self, score, sigma: float=1.0) -> np.ndarray:
        """
        Convert a float score (3 ~ 10) to a Gaussian based soft label (length = 8)
        
        Args:
            score: float in range [3, 10]
            sigma: standard deviation of Gaussian kernel (controls smoothness)
        """

        # Type cast for safety
        score = float(score)

        # Class centers: [3, 4, ..., 10]
        class_centers = np.linspace(3, 10, self.num_classes)

        # Compute Gaussian probabilities
        label = np.exp(- (score - class_centers) ** 2 / (2 * sigma ** 2))

        # Normalize
        label /= np.sum(label)

        return label.astype(np.float32)

    def score_to_soft_label(self, score) -> np.ndarray:
        """
        Convert a float score (3 ~ 10) to a soft label (length = 8)
        using linear interpolation.
        """
        num_classes = self.num_classes 

        # Convert [3, 10] to [0, 7]
        score = score - 3

        # Clip score to valid value
        score = np.clip(score, 0, 7)
        floor = int(np.floor(score))
        ceil = min(floor + 1, num_classes)

        # Create zero-vector [0.0, 0.0, ... , 0.0]
        label = np.zeros(num_classes).astype(np.float32)

        # Get decimal part
        delta = score - floor
        label[floor - 1] = 1.0 - delta
        if ceil <= num_classes:
            label[ceil - 1] = delta

        return label

class DatasetWrapperForManualScoreRegressionWithMultiTask(DatasetWrapper):
    """
    DatasetWrapper class for `manual_score` regression

    This class used for `manual_score` prediction by regression 

    The dataset will have the shape 

    ```
    (image, {
        'manual_score': "(8,), Soft label vector.",
        'color_lighting_score': "(8,), Soft label vector.",
        'costume_detail_score': "(8,), Soft label vector.", 
        'proportion_score': "(8,), Soft label vector."
    })
    ```

    The input data should have the shape 

    ```
    ({
        `image_path`: "str, absolute path to image", 
        'manual_score': "float32, [3, 10]",
        'color_lighting_score': "float32, [3, 10]",
        'costume_detail_score': "float32, [3, 10]",
        'proportion_score': "float32, [3, 10]"
    })
    ```
    """

    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True, 
    ):
        """
        The argument `data` must contain the keys `image_path` and `manual_score`
        """

        self.inputs = {
            'image_path': data['image_path'],
            'manual_score': data['manual_score'], 
            'color_lighting_score': data['color_lighting_score'],
            'costume_detail_score': data['costume_detail_score'], 
            'proportion_score': data['proportion_score']
        } if data is not None else None

        self.width = width
        self.height = height

        self.normalize = normalize

    def load_image_for_map(self, data_slice):
        
        image = self.load_image(data_slice['image_path'])
        
        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['manual_score'], 
                data_slice['color_lighting_score'],
                data_slice['costume_detail_score'],
                data_slice['proportion_score'])
    
    def map_labels(self, image, manual_score, color_lighting_score, costume_detail_score, proportion_score):
        image, manual, color_lighting, costume_detail, proportion = tf.py_function(
            self.map_labels_py,
            (image, manual_score, color_lighting_score, costume_detail_score, proportion_score),
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        # Set shape for safety
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        # Just float value
        manual.set_shape([1])
        color_lighting.set_shape([1])
        costume_detail.set_shape([1])
        proportion.set_shape([1])

        # packing to dict again
        return (image, {
            'manual_score': manual, 
            'color_lighting_score': color_lighting, 
            'costume_detail_score': costume_detail,
            'proportion_score': proportion
        })

    def map_labels_py(self, image, manual_score, color_lighting_score, costume_detail_score, proportion_score):
        
        # Just type cast
        manual = np.array([manual_score]).astype(np.float32)

        color_lighting = np.array([color_lighting_score]).astype(np.float32)

        costume_detail = np.array([costume_detail_score]).astype(np.float32)

        proportion = np.array([proportion_score]).astype(np.float32)

        return (image, manual, color_lighting, costume_detail, proportion)
    

class DatasetWrapperForScoreClassification(DatasetWrapper):
    """
    DatasetWrapper class for Score classification

    This class using for score prediction by classification

    The dataset will have the shape 

    ```
    (image, {
        'score_prediction': "(N,), Soft label vector. N is num_classes"
        })
    ```
        
    The input data shuold have the shape 
    ({
        'image_path': "str, absolute path to image",
        'score_prediction': "float32"
    })

    """

    def __init__(
            self, data: dict,
            width: int, height: int, num_classes: int = 100, normalize: bool=True
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

        self.num_classes = num_classes

        self.normalize = normalize

    def load_image_for_map(self, data_slice):
        
        image = self.load_image(data_slice['image_path'])
        
        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, 
                data_slice['score_prediction'])
    
    def map_labels(self, image, score_value):
        image, score_vector = tf.py_function(
            self.map_labels_py,
            (image, score_value),
            (tf.float32, tf.float32)
        )

        # Set shape for safety
        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image
        score_vector.set_shape([self.num_classes]) # (N,), soft label vector

        # packing to dict again
        return (image, {
            'score_prediction': score_vector
        })

    def map_labels_py(self, image, score_value):
        
        # Convert score to soft label vector 
        # (num_classes,) like [0, 0, ..., 0.4, 0.6, ...]
        score = self.score_to_soft_label(score_value, self.num_classes)

        return (image, score)

    def score_to_soft_label(self, score, num_classes: int = 100) -> np.ndarray:
        """
        Convert a float score (1~100) to a soft label (length = num_classes)
        using linear interpolation.
        """
        score = np.clip(score, 1.0, num_classes)
        floor = int(np.floor(score))
        ceil = min(floor + 1, num_classes)
        label = np.zeros(num_classes).astype(np.float32)

        delta = score - floor
        label[floor - 1] = 1.0 - delta
        if ceil <= num_classes:
            label[ceil - 1] = delta

        return label
        
class DatasetWrapperForRatingClassification(DatasetWrapper):
    """
    DatasetWrapper class for dataset with metadata. 

    The dataset will have the shape 

    ```
    (image, {
        'rating_prediction': "0 or 1. if 'R-18' then 1 else 0"
        })
    ```
        
    The input data shuold have the shape 
    ({
        'image_path': "str, absolute path to image",
        'rating_prediction': "string, R-18 or other"
    })

    """
    def __init__(
            self, data: dict,
            width: int, height: int, num_classes: int = 100, normalize: bool=True
    ):  
        super().__init__(None, width, height, normalize)

        self.inputs = {
            'image_path': data['image_path'],
            'rating_prediction': data['rating_prediction']
        }

    def load_image_for_map(self, data_slice):

        image = self.load_image(data_slice['image_path'])

        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, data_slice['rating_prediction'])
    
    def map_labels(self, image, rating_value):
        image, rating_vector = tf.py_function(
            self.map_labels_py,
            (image, rating_value),
            (tf.float32, tf.float32)
        )

        rating_vector.set_shape([1]) # just float32

        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        return (image, {
            'rating_prediction': rating_vector
        })

    def map_labels_py(self, image, rating_value):
        
        rating_vector = 1.0 if rating_value == 'R-18' else 0.0

        rating_vector = np.array([rating_vector]).astype(np.float32)

        return (image, rating_vector)

class DatasetWrapperForSanityLevelClassification(DatasetWrapper):
    """
    DatasetWrapper class for predict `sanity_level` by Gaussian soft-label based

    The dataset will have the shape 

    ```
    (image, {
        'sanity_level': "(4,), Gaussian soft-label vector"
        })
    ```
        
    The input data shuold have the shape 
    ({
        'image_path': "str, absolute path to image",
        'sanity_level': "int, 2 or 4 or 6 or 8. 8 is R-18, 6 is NSFW, 4 is sensitive, 2 is SFW"
    })

    """
    def __init__(
            self, data: dict,
            width: int, height: int, normalize: bool=True, 
            gaussian_sigma: float=0.7
    ):  
        super().__init__(None, width, height, normalize)

        self.inputs = {
            'image_path': data['image_path'],
            'sanity_level': data['sanity_level']
        }

        self.sigma = gaussian_sigma


    def load_image_for_map(self, data_slice):

        image = self.load_image(data_slice['image_path'])

        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, data_slice['sanity_level'])
    
    def map_labels(self, image, sanity_level):
        image, sanity_vector = tf.py_function(
            self.map_labels_py,
            (image, sanity_level),
            (tf.float32, tf.float32)
        )

        sanity_vector.set_shape([4]) # Soft-label Vector

        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        # packing to dict again
        return (image, {
            'sanity_level': sanity_vector
        })

    def map_labels_py(self, image, sanity_level):

        # Convert to Gaussian vector
        sanity_vector = self.score_to_gaussian_soft_label(sanity_level, self.sigma)

        return (image, sanity_vector)
    
    def score_to_gaussian_soft_label(self, score, sigma: float=1.0) -> np.ndarray:
        """
        Convert a `sanity_level`(2 | 4 | 6 | 8) to a Gaussian based soft label (length = 4)
        
        Example: 2 -> [0.7, 0.2, 0.1, 0]

        Args:
            score: integer (2 | 4 | 6 | 8)
            sigma: standard deviation of Gaussian kernel (controls smoothness)
        """

        # Type cast for safety
        score = float(score)

        # Class centers: [2, 4, 6, 8]
        class_centers = np.linspace(2, 8, 4)

        # Compute Gaussian probabilities
        label = np.exp(- (score - class_centers) ** 2 / (2 * sigma ** 2))

        # Normalize
        label /= np.sum(label)

        return label.astype(np.float32)

class DatasetWrapperForAiClassification(DatasetWrapper):
    """
    DatasetWrapper class for dataset with metadata. 

    The dataset will have the shape 

    ```
    (image, {
        'ai_prediction': "0 or 1. if AI then 1 else 0"
        })
    ```
        
    The input data shuold have the shape 
    ({
        'image_path': "str, absolute path to image",
        'ai_prediction': "0 or 1. if AI then 1 else 0"
    })

    """
    def __init__(
            self, data: dict,
            width: int, height: int, num_classes: int = 100, normalize: bool=True
    ):  
        super().__init__(None, width, height, normalize)

        self.inputs = {
            'image_path': data['image_path'],
            'ai_prediction': data['ai_prediction']
        }

    def load_image_for_map(self, data_slice):

        image = self.load_image(data_slice['image_path'])

        # Return tuple. Because tf.py_tunction not allow using dict
        return (image, data_slice['ai_prediction'])   
      
    def map_labels(self, image, rating_value):
        image, ai_vector = tf.py_function(
            self.map_labels_py,
            (image, rating_value),
            (tf.float32, tf.float32)
        )

        ai_vector.set_shape([1]) # just float32

        image.set_shape([self.height, self.width, 3]) # (height, width, channel) image

        return (image, {
            'ai_prediction': ai_vector
        })

    def map_labels_py(self, image, ai_value):
        
        ai_vector = np.array([ai_value]).astype(np.float32)

        return (image, ai_vector)
    
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