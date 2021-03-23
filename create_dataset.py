import numpy as np
from networkx import gnm_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from random_walks import CTRW, CTQRW
from config import config

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

ctqrw = CTQRW()

advantages = []

target = config['target_node']

for n in range(2, config['nodes']):
    pt = np.log(n)
    m = np.random.randint(low=n - 1, high=(n ** 2 - n) / 2)

    for _ in range(3):
        G = gnm_random_graph(n, m, seed = RANDOM_SEED)
        adj = adjacency_matrix(G).todense()

        # Quantum times
        hist = ctqrw.run(time_steps = config['number_of_steps'],
                         adjacency_matrix = adj,
                         initial_node = config['initial_node'],
                         target_node = target)
        q_t = 0
        for state_num, state in enumerate(hist.states):
            if state.full()[target, target] > pt:
                q_t = state_num
