import tensorflow as tf
import numpy as np
from tensorflow import keras


class EdgeToEdge(keras.layers.Layer):
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


def model(input):
    model = tf.keras.models.Sequential([
        EdgeToEdge(n=input.shape[1]),
        EdgeToVertex(n=input.shape[1]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='sgd',
                  loss=['binary_crossentropy'],
                  metrics=['accuracy', 'Recall', 'Precision', 'AUC'])

    return model