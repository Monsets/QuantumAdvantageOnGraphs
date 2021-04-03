import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from config import config
from utils import make_dirs
from graph_generation import generate_path_graphs, generate_random_graphs

from time import time

if __name__ == '__main__':
    make_dirs()

    np.random.seed(config['random_seed'])

    ''' Path graph with nodes [2, 10]'''

    start = time()

    if config['include_path_graphs']:
        generate_path_graphs()

    if config['include_random_graphs']:
        generate_random_graphs()

    print('Total time {}'.format(time() - start))


