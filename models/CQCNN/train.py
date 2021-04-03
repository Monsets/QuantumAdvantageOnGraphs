import os
import numpy as np
from models.CQCNN.CQCNN import CQNN
from models.CQCNN.dataloader import DataGenerator
from utils import plot_save_metrics
from config import config

GRAPH_TRAIN_PATH = os.path.join('../../', config['data_path'], 'random_graphs1', 'train', config['adjacency_path'])
RANK_TRAIN_PATH = os.path.join('../../', config['data_path'], 'random_graphs1', 'train',  config['rank_path'])

GRAPH_TEST_PATH = os.path.join('../../', config['data_path'], 'random_graphs1', 'test', config['adjacency_path'])
RANK_TEST_PATH = os.path.join('../../', config['data_path'], 'random_graphs1', 'test',  config['rank_path'])

train_generator = DataGenerator(graph_path = GRAPH_TRAIN_PATH,
                                label_path = RANK_TRAIN_PATH,
                                batch_size = 1)

test_generator = DataGenerator(graph_path = GRAPH_TEST_PATH,
                                label_path = RANK_TEST_PATH,
                                batch_size = 1)

cqnn = CQNN(input = 25)
hist = cqnn.fit(train_generator,
         validation_data=test_generator,
         epochs=config['cqnn_epochs'],
         steps_per_epoch=config['cqnn_epochs'], validation_freq=100, validation_steps=test_generator.__len__(), verbose = 1)


print(cqnn.evaluate(test_generator))
print(np.sum(np.argmax(cqnn.predict_generator(test_generator), axis = 1)))
hist = hist.history
plot_save_metrics(hist)
