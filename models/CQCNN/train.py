import os
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf

from models.CQCNN.CQCNN import CQNN
from models.CQCNN.dataloader import DataGenerator
from utils import plot_save_metrics
from config import config
from sklearn.metrics import classification_report

GRAPH_TRAIN_PATH = os.path.join('../../', config['data_path'], 'random_graphs', 'train', config['adjacency_path'])
RANK_TRAIN_PATH = os.path.join('../../', config['data_path'], 'random_graphs', 'train',  config['rank_path'])

GRAPH_TEST_PATH = os.path.join('../../', config['data_path'], 'random_graphs', 'test', config['adjacency_path'])
RANK_TEST_PATH = os.path.join('../../', config['data_path'], 'random_graphs', 'test',  config['rank_path'])

train_generator = DataGenerator(graph_path = GRAPH_TRAIN_PATH,
                                label_path = RANK_TRAIN_PATH,
                                batch_size = 3)

test_generator = DataGenerator(graph_path = GRAPH_TEST_PATH,
                                label_path = RANK_TEST_PATH,
                                batch_size = 3)

ys = []

for _, y in train_generator:
    ys.extend(y)
for _, y in test_generator:
    ys.extend(y)
print(np.bincount(np.argmax(ys, axis = 1)))

cqnn = CQNN(input = 15)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models/CQCNN/weights',
                                                 save_weights_only=True,
                                                 verbose=1)

avg_callback = tfa.callbacks.AverageModelCheckpoint(filepath='models/CQCNN/weights',
                                                    update_weights=True)

hist = cqnn.fit(train_generator,
         validation_data=test_generator,
         epochs=config['cqnn_epochs'],
         steps_per_epoch=config['cqnn_samples_per_epoch'], validation_freq=300, validation_steps=test_generator.__len__(), verbose = 300)


preds = np.argmax(cqnn.predict_generator(test_generator), axis = 1)
y_true = []
for x, y in test_generator:
    y_true.extend(y)

print(classification_report(y_true = np.argmax(y_true, axis = 1),
                            y_pred = preds))
hist = hist.history
# plot_save_metrics(hist)
