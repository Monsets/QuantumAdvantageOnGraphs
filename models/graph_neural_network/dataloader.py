import copy
import os
import pickle
import numpy as np
import networkx as nx

from graph_nets import utils_np


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class GraphTypleDataLoader:

    def __init__(self, filepath, batch_size, use_only=False, shuffle=True, adj_name='adjacency.pkl',
                 node_features='node_features.pkl',
                 edge_features='edge_features.pkl', y_name='y.pkl'):

        # Inner index for iterating through graphs
        self.indx = 0
        self.batch_size = batch_size

        # Reading files
        adjacency_matrices = read_pickle(os.path.join(filepath, adj_name))
        node_feature_matrices = read_pickle(os.path.join(filepath, node_features))

        self._build_networkx_graphs(adjacency_matrices, node_feature_matrices)
        self.y_matrices = read_pickle(os.path.join(filepath, y_name))

        self.len = len(self.y_matrices)
        if use_only > self.len:
            raise IndexError('{} is out of data len which is {}'.format(use_only, self.len))
        if use_only:
            self.len = use_only
            self.indices = np.array(range(use_only))
        else:
            self.indices = np.array(range(self.len))

        self.steps = len(self.indices) // batch_size
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next(self, raw_graphs=False):
        if self.indx + self.batch_size > self.len:
            self.indx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        indx_range = self.indices[range(self.indx, self.indx + self.batch_size)]

        graphs = [self.graphs[i] for i in indx_range]
        ys = [self.y_matrices[i] for i in indx_range]
        targets = copy.deepcopy(graphs)
        for i in range(len(ys)):
            graphs[i].graph['features'] = np.array([0, 0]).astype(np.float)
            targets[i].graph['features'] = np.array(ys[i]).astype(np.float)

        self.indx += self.batch_size

        if raw_graphs == True:
            return graphs, targets
        else:
            return utils_np.networkxs_to_graphs_tuple(graphs), utils_np.networkxs_to_graphs_tuple(targets)

    def _build_networkx_graphs(self, adjacency_matrices, node_feature_matrices):
        self.graphs = []

        for adj, feat in zip(adjacency_matrices, node_feature_matrices):
            nx_graph = nx.from_numpy_matrix(np.array(adj))

            # Set node features
            for i, node_feature in enumerate(feat):
                nx_graph.nodes[i]['features'] = np.array(node_feature).astype(np.float).reshape(1)

            # Set edge features to None if don't do that excplicitly conversion to graph_tuple fails
            for edge in nx_graph.edges:
                nx_graph.edges[edge]['features'] = np.array(0).astype(np.float).reshape(1)

            self.graphs.append(copy.deepcopy(nx_graph))
            del nx_graph