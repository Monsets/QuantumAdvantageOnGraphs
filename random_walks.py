import numpy as np
import qutip as Q

from scipy.constants import hbar
from scipy.linalg import expm

class CTRW():
    def __init__(self):
        pass

    def run(self, time_steps, adjacency_matrix, initial_node, target_node):
        '''
        Simulates Continious-time random walk for given time_steps and adjacency matrix.
        Transition matrix is created using node-centric approach with equiprobable transition
        for each adjacent node.
        :param time_steps: Number of steps for which evolution of matrix will be calculated
        :param adjacency_matrix: Adjacency matrix of a graph in time_step = 0
        :param initial_node: Initial node
        :param target_node: Target node which will be used as an absorbing node

        :return: history of vectors with probabilities of states
        '''

        # Initialize starting states
        len_nodes = len(adjacency_matrix)
        # Initial vector of states with initial node
        initial_vector = np.zeros((len_nodes))
        initial_vector[initial_node] = 1
        initial_vector = initial_vector

        # Create node-centric transition matrix
        transition_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis = 0)
        # Absorbing node
        transition_matrix[:, target_node] = 0
        transition_matrix[target_node, target_node] = 1
        # Collects states of probability vector
        vector_history = []

        dt = 0.1
        T = transition_matrix
        # Apply differential equation to obtain probabilities of states
        for step in np.arange(1, time_steps + 1, dt):
            t = dt * step
            vector = np.exp(-t) * np.matmul(expm((T * t)), initial_vector)
            vector_history.append(vector)


        return vector_history

class CTQRW():
    def __init__(self):
        pass

    def run(self, time_steps, adjacency_matrix, initial_node, target_node, dt = 0.01):
        '''
        Simulates Continious-time random walk for given time_steps and adjacency matrix.
        Transition matrix is created using node-centric approach with equiprobable transition
        for each adjacent node.
        :param time_steps: Number of steps for which evolution of matrix will be calculated
        :param adjacency_matrix: Adjacency matrix of a graph in time_step = 0
        :param initial_node: Initial node
        :param target_node: Target node which will be used as an absorbing node

        :return: history of vectors with probabilities of states
        '''

        # Initialize starting states
        len_nodes = len(adjacency_matrix)

        # Create transition matrix with sink node
        transition_matrix = np.concatenate((adjacency_matrix, np.zeros((1, len_nodes))))
        transition_matrix = np.concatenate((transition_matrix, np.zeros((len_nodes + 1, 1))), axis=1)
        H = Q.Qobj(np.array(transition_matrix).astype(float))
        len_nodes += 1

        # Apply differential equation to obtain probabilities of states
        times = np.arange(0, time_steps, dt)

        # Creating jump function L |sink><target|
        sink = np.zeros((1, len_nodes)).T
        sink[-1, 0] = 1
        target = np.zeros((len_nodes , 1))
        target[target_node, 0] = 1
        # L - [num_nodes, num_nodes] matrix
        L = Q.Qobj(np.outer(sink, target))

        # Creating basis
        initial = np.zeros((len_nodes, 1))
        initial[initial_node, 0] = 1
        # L - [num_nodes, num_nodes] matrix
        p0 = Q.Qobj(np.outer(initial.T, initial))

        vectors = Q.mesolve(H, p0, times, [L], [])

        return vectors

ctrw = CTRW()
hist = ctrw.run(100, np.array([[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]]), 0, 1)
for i, vector in enumerate(hist):
    print("Time {} Vector {}".format(i, vector))


# ctqrw = CTQRW()
# hist = ctqrw.run(5, np.array([[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]]), 0, 1)
