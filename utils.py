import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt

from config import config


def make_dirs():
  if not os.path.exists(config['data_path']):
    os.mkdir(config['data_path'])

  for s in [config['path_graph_path'], config['path_random_path']]:

    train_path = os.path.join(config['data_path'], s, 'train')
    test_path = os.path.join(config['data_path'], s, 'test')

    if not os.path.exists(train_path):
      os.mkdir(train_path)

    if not os.path.exists(test_path):
      os.mkdir(test_path)


def save_graph_rank(graph, rank, node_features, path):
  '''
  Saves graph with pickle
  :param graph - adjacency matrix to save
  :param rank - y in the shape [classical, quantum]
  :param node_features - array of node features
  :param path - path to save all
  '''
  g_path = os.path.join(path, config['adjacency_path'])
  r_path = os.path.join(path, config['rank_path'])
  n_path = os.path.join(path, config['node_feature_path'])

  with open(n_path, 'wb') as f:
    pickle.dump(node_features, f)

  with open(g_path, 'wb') as f:
    pickle.dump(graph, f)

  with open(r_path, 'wb') as f:
    pickle.dump(rank, f)

def plot_save_metrics(hist, path = 'results'):
  plt.subplot(2, 2, 1)
  plt.plot(hist['accuracy'], label='train accuracy')
  plt.plot(hist['val_accuracy'], label='val accuracy')
  plt.legend(loc='upper left')
  plt.title('Accuracy')

  plt.subplot(2, 2, 2)
  plt.plot(hist['recall'], label='train recall')
  plt.plot(hist['val_recall'], label='val recall')
  plt.legend(loc='upper left')
  plt.title('Recall')

  plt.subplot(2, 2, 1)
  plt.plot(hist['precision'], label='train precision')
  plt.plot(hist['val_precision'], label='val precision')
  plt.legend(loc='upper left')
  plt.title('Precision')

  plt.subplot(2, 2, 1)
  plt.plot(hist['loss'], label='train loss')
  plt.plot(hist['val_loss'], label='val loss')
  plt.legend(loc='upper left')
  plt.title('Loss')

  # plt.imsave(path + '.png')
  plt.show()
