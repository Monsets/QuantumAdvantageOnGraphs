import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow_addons.optimizers import SWA

''' Layers'''

class EdgeToEdge(keras.layers.Layer):
    """
        Edge to edge layer
    """
    def __init__(self, n):

        super(EdgeToEdge, self).__init__()

        K = np.zeros((n + 1, n + 1))
        K[:, n // 2] = 1
        K[n // 2, :] = 1
        K[n // 2, n // 2] = 0

        self.K = tf.constant(K.reshape(n + 1, n + 1, 1, 1), dtype=float)

    def call(self, inputs):
        out = tf.nn.conv2d(
            inputs, self.K, padding='SAME', strides=1
        )
        return out * inputs


class EdgeToVertex(keras.layers.Layer):
    """
        Edge to edge layer
    """
    def __init__(self, n):
        super(EdgeToVertex, self).__init__()
        K = np.zeros((n + 1, n + 1))
        K[:, n // 2] = 1
        K[n // 2, :] = 1
        K[n // 2, n // 2] = 0

        self.K = tf.constant(K.reshape(n + 1, n + 1, 1, 1), dtype=float)
        self.n = n

    def call(self, inputs):
        input = tf.linalg.band_part(inputs[:, :, :, 0], -1, 0)
        input = tf.expand_dims(input, -1)

        out = tf.nn.conv2d(
            input, self.K, padding='SAME', strides=(self.n, 1)
        )
        return out

def custom(y_true, y_pred):
    cost = tf.math.log(tf.math.exp(y_true) / (tf.math.exp(y_pred[:, 0]) + tf.math.exp(y_pred[:, 1])) )

def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits,
    # Keras expects probabilities.
    # transform y_pred back to logits


    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight= 1949 / (1949 + 49))

    cost = tf.reduce_mean(cost)

    return cost

''' Model '''

def CQNN(input):
    model = tf.keras.models.Sequential([
        EdgeToEdge(n=input),
        tf.keras.layers.Conv2D(10, 3, 1, 'same'),
        EdgeToVertex(n=input),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    sgd = tf.keras.optimizers.SGD(0.01)
    # stocastic_avg_sgd = SWA(sgd, )

    model.compile(optimizer=sgd,
                  loss=cross_entropy_balanced,
                  metrics=['accuracy', 'Recall', 'Precision', 'AUC'])

    return model