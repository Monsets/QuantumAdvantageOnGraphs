import numpy as np

from models.graph_neural_network.model import GNNModel
from utils import plot_metrics_per_class

if __name__ == '__main__':
    seed = 2
    rand = np.random.RandomState(seed=seed)
    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 5
    num_processing_steps_ge = 5
    epochs = 1000

    batch_size = 64
    USE_ONLY = 3000
    PATH = 'data/random_graphs'

    backbone = 'en'
    model = GNNModel(backbone = backbone,
                    datapath = PATH,
                    batch_size = batch_size,
                    message_passing_steps_train = num_processing_steps_tr,
                    epochs = epochs)

    history = model.train()

    plot_metrics_per_class(history, '')
