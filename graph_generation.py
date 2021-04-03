import os

import numpy as np

from sklearn.model_selection import train_test_split

from config import config
from random_walks import CTRW, CTQRW
from itertools import permutations
from networkx import path_graph, adjacency_matrix, gnm_random_graph

from utils import save_graph_rank


def predict_advantage(adj, initial_node, target_node):
  n = len(adj)
  # Probability threshold - log of number of nodes
  pt = 1 / np.log(n)

  # Classical walk
  hist = CTRW().run(time_steps=config['number_of_steps'],
                    adjacency_matrix=adj,
                    initial_node=initial_node,
                    target_node=target_node,
                    dt=config['dt'])
  # Get iteration number
  c_t = 0
  for vector_num, vector in enumerate(hist):
    if vector[target_node] > pt:
      c_t = vector_num
      break

  # Quantum walkLinear
  hist = CTQRW().run(time_steps=config['number_of_steps'],
                     adjacency_matrix=adj,
                     initial_node=initial_node,
                     target_node=target_node,
                     dt=config['dt'])
  # Get iteration number
  q_t = 0
  if type(hist) != list:
    for state_num, state in enumerate(hist.states):
      # Collect probabilities from the sink node
      if state.full()[-1, -1] > pt:
        q_t = state_num
        break

  # [c_t, q_t]
  rank = np.zeros(2)


  # In case if both walk failed don't take a sample
  if c_t == 0 and q_t == 0:
    return []
  # Which walk is faster
  if 0 < q_t < c_t or q_t > 0 and c_t == 0:
    rank[1] = 1
  else:
    rank[0] = 1



  return rank

def generate_path_graphs():
  '''
  Generates and saves path graphs with number of vertices number in [3, config['max_node_for_path_graph]
  '''

  graphs = []
  ranks = []
  node_features = []

  for n in range(3, config['max_node_for_path_graph']):
    # Path graph stays the same. The difference between graphs are only special nodes number
    # and those might be expressed through permutations
    print(n)
    perms = list(permutations(range(n), n - 1))

    for perm_num, perm in enumerate(perms):
      # Verbose
      if perm_num % 10 == 0:
        print('{} / {}'.format(perm_num, len(perms)))
      # Permutations are possible for n - 1 list, so we add the number that is not in the list
      perm = list(perm)
      perm.extend([numb for numb in range(n) if numb not in perm])

      # Generate graph and adjacency matrix
      g = path_graph(perm)
      adj = adjacency_matrix(g).todense()
      # Calculate initial and target node according to the permutation
      initial_node = np.where(np.array(perm) == config['initial_node'])[0][0]
      target_node = np.where(np.array(perm) == config['target_node'])[0][0]


      # Node features are starting and ending nodes
      node_feature = np.zeros(len(perm))
      node_feature[initial_node] = 1
      node_feature[target_node] = 1

      rank = predict_advantage(adj, initial_node, target_node)

      if len(rank) < 1:
        continue

      graphs.append(adj)
      ranks.append(rank)
      node_features.append(node_feature)


  ranks = np.array(ranks)


  ''' Describing dataset '''
  print('Classical wins {} Quantum wins {} '.format(np.sum(ranks[:, 0] == True),
                                                    np.sum(ranks[:, 1] == True), ))

  ''' Split and Save dataset '''
  g_train, g_test, n_f_train, n_f_test, y_train, y_test = \
    train_test_split(graphs, node_features, ranks)

  save_graph_rank(g_train, y_train, n_f_train, os.path.join(config['data_path'],
                                                            config['path_graph_path'], 'train'))
  save_graph_rank(g_test, y_test, n_f_test, os.path.join(config['data_path'],
                                                         config['path_graph_path'], 'test'))


def generate_random_graphs():
  ''' Random graphs '''

  graphs = []
  ranks = []
  node_features = []

  # Generate number of nodes for random graphs

  vertex_number = np.random.randint(low=config['min_node_for_random_graph'],
                                    high=config['max_node_for_random_graph'],
                                    size=config['random_graphs_amount'])

  for step, n in enumerate(vertex_number):
    if step % 50 == 0:
      print(' {} / {} '.format(step, len(vertex_number)))

    m = np.random.randint(low=n - 1, high=(n ** 2 - n) // 2)

    g = gnm_random_graph(n=n, m=m, seed=config['random_seed'])

    adj = adjacency_matrix(g).todense()

    # Node features are starting and ending nodes
    # Encoded as vectors [true if initial node, true if target node]
    node_feature = np.zeros((n, 2))
    node_feature[config['initial_node'], 0] = 1
    node_feature[config['target_node'], 1] = 1

    rank = predict_advantage(adj, config['initial_node'], config['target_node'])

    if len(rank) < 1:
      continue

    graphs.append(adj)
    ranks.append(rank)
    node_features.append(node_feature)

  ranks = np.array(ranks)


  ''' Describing dataset '''
  print('Classical wins {} Quantum wins {} '.format(np.sum(ranks[:, 0] == True),
                                                    np.sum(ranks[:, 1] == True)))

  ''' Saving dataset '''
  g_train, g_test, n_f_train, n_f_test, y_train, y_test = \
    train_test_split(graphs, node_features, ranks, test_size=0.1)

  save_graph_rank(g_train, y_train, n_f_train, os.path.join(config['data_path'],
                                                            config['path_random_path'], 'train'))
  save_graph_rank(g_test, y_test, n_f_test, os.path.join(config['data_path'],
                                                         config['path_random_path'], 'test'))