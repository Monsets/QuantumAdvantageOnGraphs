import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import json

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
  print(hist)
  plt.plot(hist['accuracy'], label='train accuracy')
  plt.plot(hist['val_accuracy'], label='val accuracy')
  plt.legend(loc='upper left')
  plt.title('Accuracy')

  plt.subplot(2, 2, 2)
  plt.plot(hist['recall'], label='train recall')
  plt.plot(hist['val_recall'], label='val recall')
  plt.legend(loc='upper left')
  plt.title('Recall')

  plt.subplot(2, 2, 3)
  plt.plot(hist['precision'], label='train precision')
  plt.plot(hist['val_precision'], label='val precision')
  plt.legend(loc='upper left')
  plt.title('Precision')

  plt.subplot(2, 2, 4)
  plt.plot(hist['loss'], label='train loss')
  plt.plot(hist['val_loss'], label='val loss')
  plt.legend(loc='upper left')
  plt.title('Loss')

  # plt.imsave(path + '.png')
  plt.show()


def plot_metrics_per_class(history, save_path):
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(history, f)
    plt.subplot(2, 2, 1)
    print(history)
    plt.plot(history['accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.legend(loc='upper left')
    plt.xticks(history['iter'])
    plt.title('Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(history['0']['recall'], label='classical train recall')
    plt.plot(history['1']['recall'], label='quantum val recall')
    plt.plot(history['0']['val_recall'], label='classical val recall')
    plt.plot(history['1']['val_recall'], label='quantum val recall')
    plt.legend(loc='upper left')
    plt.xticks(history['iter'])

    plt.title('Recall')

    plt.subplot(2, 2, 3)
    plt.plot(history['0']['precision'], label='classical train precision')
    plt.plot(history['1']['precision'], label='quantum val precision')
    plt.plot(history['0']['val_precision'], label='classical val precision')
    plt.plot(history['1']['val_precision'], label='quantum val precision')
    plt.legend(loc='upper left')
    plt.xticks(history['iter'])

    plt.title('Precision')

    plt.subplot(2, 2, 4)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.legend(loc='upper left')
    plt.title('Loss')
    plt.xticks(history['iter'])

    plt.savefig(os.path.join(save_path, 'metrics.png'))