import tensorflow as tf

class MacroPrecision(tf.keras.metrics.Metric):
    """
    Custom metrics for multi class classification
    """
    
    def __init__(self, num_classes, name="macro_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Add weight manually because cannot use assign_add to SymbolicTensor
        self.true_positives = [
            self.add_weight(name=f"tp_{c}", initializer="zeros") for c in range(num_classes)
        ]
        self.false_positives = [
            self.add_weight(name=f"fp_{c}", initializer="zeros") for c in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        for c in range(self.num_classes):
            y_true_c = tf.cast(tf.equal(y_true, c), self.dtype)
            y_pred_c = tf.cast(tf.equal(y_pred, c), self.dtype)
            tp = tf.reduce_sum(y_true_c * y_pred_c)
            fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
            self.true_positives[c].assign_add(tp)
            self.false_positives[c].assign_add(fp)

    def result(self):
        precisions = []
        for tp, fp in zip(self.true_positives, self.false_positives):
            precisions.append(tp / (tp + fp + 1e-8))
        return tf.reduce_mean(tf.stack(precisions))

    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class MacroRecall(tf.keras.metrics.Metric):
    """
    Custom metrics for multi class classification
    """
    
    def __init__(self, num_classes, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = [
            self.add_weight(name=f"tp_{c}", initializer="zeros") for c in range(num_classes)
        ]
        self.false_negatives = [
            self.add_weight(name=f"fn_{c}", initializer="zeros") for c in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        for c in range(self.num_classes):
            y_true_c = tf.cast(tf.equal(y_true, c), self.dtype)
            y_pred_c = tf.cast(tf.equal(y_pred, c), self.dtype)
            tp = tf.reduce_sum(y_true_c * y_pred_c)
            fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))
            self.true_positives[c].assign_add(tp)
            self.false_negatives[c].assign_add(fn)

    def result(self):
        recalls = []
        for tp, fn in zip(self.true_positives, self.false_negatives):
            recalls.append(tp / (tp + fn + 1e-8))
        return tf.reduce_mean(tf.stack(recalls))

    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)