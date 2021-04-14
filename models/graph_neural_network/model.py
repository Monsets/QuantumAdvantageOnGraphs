import os

import tensorflow as tf
import numpy as np

from graph_nets import utils_tf, utils_np
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

from models.graph_neural_network1.backbones import EncodeProcessDecode
from models.graph_neural_network1.dataloader import GraphTypleDataLoader


class Model:

    def __init__(self, backbone, datapath, batch_size, message_passing_steps_train,
                 learning_rate = 1e-3, verbose = 5, epochs = 1000,
                 message_passing_steps_test = None, use_only = False,):

        '''

        :param backbone: Which backbone for model to use allowed: (en - EncoderProcessDecode)
        :param datapath: Path to data with graphs with two folders for train and test
        :param batch_size: Number of graphs to run in one iteration
        :param message_passing_steps_train: Number of message-passing-steps for training
        :param learning_rate: Learning rate for
        :param verbose: Period of logging (every {verbose} epoch)
        :param epochs: Number of epochs to train the model
        :param message_passing_steps_test: Number of message-passing-steps for test if None equals to training steps
        :param use_only: If passed uses only passed amount of graphs from the data. Splits 90% of a number for train and 10% for test
        '''

        if backbone == 'en':
            self.model = EncodeProcessDecode(edge_output_size=2, node_output_size=2, global_output_size=2)

        # Initialize loss function
        self._create_loss_ops()
        self.datapath = datapath
        self.use_only = use_only
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.message_passing_steps_train = message_passing_steps_train

        if not message_passing_steps_train:
            self.message_passing_steps_test = message_passing_steps_train
        else:
            self.message_passing_steps_test = message_passing_steps_test

    def _create_loss_ops(self, output_ops):
        loss_ops = [
            tf.losses.softmax_cross_entropy(self.target_ph.globals, output_op.globals)
            for output_op in output_ops
        ]

        return loss_ops

    def _create_placeholders(self, loader):
        '''
        Creates input and output placeholders for model. Loader is used to get the shapes of the data
        :param loader: GraphTypleDataLoader instance. Used only to get the shapes of the graphs.
        '''
        # Create some example data for inspecting the vector sizes.
        input_graphs, target_graphs = loader.next(raw_graphs=True)
        self.input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
        self.target_ph = utils_tf.placeholders_from_networkxs(target_graphs)


    def _make_all_runnable_in_session(*args):
        """Lets an iterable of TF graphs be output from a session as NP graphs."""
        return [utils_tf.make_runnable_in_session(a) for a in args]

    def _create_feed_dict(self, dataloader):
        """Creates placeholders for the model training and evaluation.
        """
        input_graphs, target_graphs = dataloader.next()
        feed_dict = {self.input_ph: input_graphs, self.target_ph: target_graphs}

        return feed_dict

    def _compute_metrics(self, target, output):
        """Calculate model accuracy, precision, recall.

        Returns list with metrics

        Args:
          target: A List of `graphs.GraphsTuple` that contains the target graph.
          output: A List of `graphs.GraphsTuple` that contains the output graph.

        Returns:
          metrics: A List with accuracy, recall and precision scores

        Raises:
          ValueError: Nodes or edges (or both) must be used
        """
        true, preds = [], []

        # Unzip list of graphs
        for t, o in zip(target, output):
            d_t = utils_np.graphs_tuple_to_data_dicts(t)
            d_o = utils_np.graphs_tuple_to_data_dicts(o)
            # Obtain target and prediction for every graph
            for td, od in zip(d_t, d_o):
                true.append(np.argmax(td['globals']))
                preds.append(np.argmax(od['globals']))

        metrics = [accuracy_score(true, preds), recall_score(true, preds), precision_score(true, preds)]

        return metrics

    def _initialize_loaders(self):
        '''
        Initializes loaders for training and testing
        '''
        if self.use_only:
            use_only_tr = int(self.use_only * .9)
            use_only_te = int(self.use_only * .1)
        else:
            use_only_tr = None
            use_only_te = None

        self.dl_train = GraphTypleDataLoader(filepath=os.path.join(self.datapath, 'train'), batch_size=self.batch_size,
                                        use_only=use_only_tr)
        self.dl_test = GraphTypleDataLoader(filepath=os.path.join(self.datapath, 'test'), batch_size=self.batch_size,
                                       use_only=use_only_te)

    def _initialize_model(self):
        '''
        Creates placeholders, loss functions, optimizers, steps and everything else that is needed for model to train
        '''
        # Here we will save losses, metrics
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
                        'recall': [], 'val_recall': [], 'precision': [], 'val_precision': [],
                        '0': {'train': [], 'val': []}, '1': {'train': [], 'val': []}}
        # Data.
        # Input and target placeholders.
        self._create_placeholders(self.dl_train)

        # Connect the data to the model.
        # Instantiate the model.
        # A list of outputs, one per processing step.
        self.output_ops_tr = self.model(self.input_ph, self.message_passing_steps_train)
        self.output_ops_ge = self.model(self.input_ph, self.message_passing_steps_test)

        # Training loss.
        self.loss_ops_tr = self._create_loss_ops(self.output_ops_tr)
        # Loss across processing steps.
        self.loss_op_tr = sum(self.loss_ops_tr) / self.message_passing_steps_train
        # Test/generalization loss.
        self.loss_ops_ge = self._create_loss_ops(self.output_ops_ge)
        self.loss_op_ge = self.loss_ops_ge[-1]  # Loss from final processing step.

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.step_op = optimizer.minimize(self.loss_op_tr)

        # Lets an iterable of TF graphs be output from a session as NP graphs.
        self._make_all_runnable_in_session()

    def train(self):
        '''

        :return:
        '''

        self._initialize_loaders()
        self._initialize_model()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.epochs):
            # Save outputs from batches to compute metrics on all data
            train_targets = []
            train_preds = []
            train_losses = []

            for step in range(self.dl_train.steps):
                # Train on batch
                feed_dict = self._create_feed_dict(self.dl_train)
                train_values = sess.run({
                    "step": self.step_op,
                    "target": self.target_ph,
                    "loss": self.loss_op_tr,
                    "outputs": self.output_ops_tr
                },
                    feed_dict=feed_dict)

                # Save outputs and targets to calculate metrics
                if epoch % self.verbose == 0:
                    train_preds.append(train_values["outputs"][-1])
                    train_targets.append(train_values["target"])
                    train_losses.append(train_values['loss'])

            # Calculate metrics for train and test if current epoch is logging
            if epoch % self.verbose == 0:
                metrics_train = self._compute_metrics(
                    train_targets, train_preds)

                # Save outputs from batches to compute metrics on all data
                test_losses = []
                test_targets = []
                test_preds = []

                # Calculate metrics for evaluation
                for val_step in range(self.dl_test.steps):
                    # Train on batch
                    feed_dict = self._create_feed_dict(
                        self.dl_test)
                    test_values = sess.run({
                        "target": self.target_ph,
                        "loss": self.loss_op_ge,
                        "outputs": self.output_ops_ge
                    }, feed_dict=feed_dict)

                    test_preds.append(test_values["outputs"][-1])
                    test_targets.append(test_values["target"])
                    test_losses.append(test_values['loss'])

                metrics_test = self._compute_metrics(
                    test_targets, test_preds)

                # Save history
                self.history['accuracy'].append(metrics_train[0])
                self.history['val_accuracy'].append(metrics_train[0])
                self.history['recall'].append(metrics_train[2])
                self.history['val_recall'].append(metrics_train[2])
                self.history['precision'].append(metrics_train[1])
                self.history['val_precision'].append(metrics_train[1])
                self.history['loss'].append(np.mean(train_losses))
                self.history['val_loss'].append(np.mean(test_losses))
                # Metrics per class
                class_report_val = classification_report(test_targets, test_preds)
                class_report = classification_report(train_targets, train_preds)
                self.history['0']['train'].append(class_report['0'])
                self.history['0']['val'].append(class_report_val['0'])
                self.history['1']['train'].append(class_report['0'])
                self.history['1']['val'].append(class_report_val['1'])

                print("# {:05d}, Train loss {:.4f}, Test loss {:.4f}, Train Accuracy {:.4f}, Test Accuracy"
                      " {:.4f}, Train Recall {:.4f}, Test Recall"
                      " {:.4f}, Train Precision {:.4f}, Test Precision"
                      " {:.4f},".format(epoch, np.mean(train_losses), np.mean(test_losses), metrics_train[0],
                                        metrics_test[0],
                                        metrics_train[1],
                                        metrics_test[1],
                                        metrics_train[2],
                                        metrics_test[2]))
        return self.history
