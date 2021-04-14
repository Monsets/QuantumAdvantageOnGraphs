import numpy as np

from models.graph_neural_network.backbones import EncodeProcessDecode
from models.graph_neural_network.model import Model
from utils import plot_save_metrics

if __name__ == '__main__':
    seed = 2
    rand = np.random.RandomState(seed=seed)
    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 2
    num_processing_steps_ge = 2
    num_training_iterations = 10000

    batch_size = 128
    USE_ONLY = 3000
    PATH = 'data/random_graphs'

    backbone = 'en'
    model = Model(backbone = backbone,
                    datapath = PATH,
                    batch_size = batch_size,
                    message_passing_steps_train = num_processing_steps_tr,
                    epochs = 1)

    history = model.train()

    plot_save_metrics(history)
