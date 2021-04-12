import tensorflow as tf
import numpy as np

import pickle

class DataGenerator(tf.keras.utils.Sequence):

  'Generates datax for keras'
  def __init__(self, graph_path, label_path, batch_size, shuffle=True):
    'Initialization'
    print(graph_path)
    with open(graph_path, 'rb') as f:
      self.data = np.array(pickle.load(f)).astype(np.float)
      print(self.data.shape)
    with open(label_path, 'rb') as f:
      self.ranks = np.array(pickle.load(f)).astype(np.float)

    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return len(self.data) // self.batch_size

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    X, Y = self.data[indexes], self.ranks[indexes]

    return tf.convert_to_tensor(X.reshape(*X.shape, 1)), tf.convert_to_tensor(Y)

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
