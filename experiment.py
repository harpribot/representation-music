import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dnn.loss import mse
from dnn.optimizer import Optimizer
from Models.low_level_sharing_model import LowLevelSharingModel
from utils.data_utils.labels import Labels
from utils.data_utils.data_handler import fetch_data, create_experiment
from utils.argument_parser import parse_arguments


class Experiment(object):
    def __init__(self, task_ids, x_train, x_validate, x_test, y_train, y_validate, y_test, model_class, expt_name,
                 learning_rate, batch_size, num_epochs, checkpoint_freq, evaluation_freq):
        """
        Class to run experiments.
        :param task_ids: List of task identifiers
        :param x_train: Training set input
        :param x_validate: Validation set input
        :param x_test: Test set input
        :param y_train: Training set labels
        :param y_validate: Validation set labels
        :param y_test: Test set labels
        :param model_class: A class derived from the Model class
        :param expt_name: Name for the experiment. This will be used as a prefix to the name of a directory created to
            store the logs and output of this experiment.
        :param learning_rate: Learning rate for SGD-based optimization.
        :param batch_size: Size of mini-batches for SGD-based training.
        :param num_epochs: Number of epochs -- full pass over the training set.
        :param checkpoint_freq: Number of mini-batch SGD steps after which an evaluation is performed.
        :param evaluation_freq: Number of mini-batch SGD steps after which a checkpoint is taken.
        :return: None
        """
        self.task_ids = task_ids
        self.x_train = x_train
        self.x_validate = x_validate
        self.x_test = x_test
        self.y_train = y_train
        self.y_validate = y_validate
        self.y_test = y_test

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_freq = checkpoint_freq
        self.evaluation_freq = evaluation_freq

        self.sess = None
        self.optimizer = None
        self.saver = None

        self.input_dimension = self.x_train.shape[1]
        self.train_samples = self.x_train.shape[0]
        self.output_dimensions = {task_id: self.y_train[task_id].shape[1] for task_id in task_ids}
        self.model = model_class(task_ids, self.input_dimension, self.output_dimensions)

        # Dictionary of list of training errors indexed by task-identifiers. Each list contains errors of the model
        # on the training set for that task over the course of training.
        self.training_errors = {}

        # Dictionary of list of validation errors indexed by task-identifiers. Each list contains errors of the model
        # on the validation set for that task over the course of training.
        self.validation_errors = {}

        self._initialize_error_dictionaries()
        create_experiment(expt_name)

    def initialize_network(self):
        """
        Initializes the DNN network
        :return: None
        """
        self.sess = tf.InteractiveSession()
        print("------")
        self.model.create_model()
        self._initialize_trainer()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def train(self):
        """
        Trains the network
        :return: None
        """
        print("------")
        print("Starting training")
        step = 0
        start_time = time.time()
        for epoch in xrange(1, self.num_epochs + 1):
            if epoch == self.num_epochs:
                print("Last epoch. All minibatches will be evaluated and checkpointed.")
            for minibatch_indices in self._iterate_minibatches(self.batch_size):
                feed_dict = dict()
                feed_dict[self.model.get_layer('input')] = self.x_train[minibatch_indices]
                for id_ in self.task_ids:
                    feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_train[id_][minibatch_indices]

                self.optimizer.run(session=self.sess, feed_dict=feed_dict)

                step += 1
                duration = int(time.time() - start_time)

                # Evaluate model fairly often, including on the last epoch.
                if step % self.evaluation_freq == 0 or epoch == self.num_epochs:
                    # Print current errors on training and validation sets.
                    t_errors = self._training_errors()
                    v_errors = self._validation_errors()
                    print("Step: {}, Epoch: {}, Duration: {}, Training Errors: {}, Validation Errors: {}"
                          .format(step, epoch, duration, t_errors, v_errors))

                    # Add current errors to the cummulative errors list for plotting.
                    for task_id in self.task_ids:
                        self.training_errors[task_id].append(t_errors[task_id])
                        self.validation_errors[task_id].append(v_errors[task_id])
                    self._plot_errors()

                # Checkpoint the model periodically, including on the last epoch
                if step % self.checkpoint_freq == 0 or epoch == self.num_epochs:
                    self.saver.save(self.sess, 'checkpoint-' + str(step).zfill(8))

    def _initialize_error_dictionaries(self):
        """
        Initialize the dictionaries for training and validation error
        :return: None
        """
        for task_id in self.task_ids:
            self.training_errors[task_id] = []
            self.validation_errors[task_id] = []

    def _initialize_trainer(self):
        """
        Initializes the training optimizer
        :return: None
        """
        self.cost = mse(0., 0.)
        for task_id in self.task_ids:
            self.cost += self.model.get_layer(task_id + '-loss')

        opt = Optimizer(self.cost)
        self.optimizer = opt.get_adagrad(self.learning_rate)

    def _iterate_minibatches(self, batch_size, shuffle=True):
        """
        Yields list of indices for each minibatch. The last mini-batch is not used because it usually has
        a different size. Since training set is shuffled before creating minibatches, the training examples in the
        ignored mini-batches are used in some other mini-batch in at least one epoch with high probability.

        :param batch_size: Size of each minibatch
        :param shuffle: Boolean to be set to True if the training set needs to be shuffled at each epoch.
        :return List of indices corresponding to a mini-batch
        """
        num_input = self.x_train.shape[0]
        indices = np.arange(num_input)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_input - batch_size + 1, batch_size):
            minibatch = indices[start_idx:start_idx + batch_size]
            yield minibatch

    def _training_errors(self):
        """
        Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the training set.

        :return dictionary of training errors
        """
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_train
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_train[id_]
        errors = {}
        for task_id in self.task_ids:
            errors[task_id] = self.model.get_layer(task_id + '-loss').eval(session=self.sess, feed_dict=feed_dict)
        return errors

    def _validation_errors(self):
        """
        Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the training set.

        :return dictionary of validation errors
        """
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_validate
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_validate[id_]
        errors = {}
        for task_id in self.task_ids:
            errors[task_id] = self.model.get_layer(task_id + '-loss').eval(session=self.sess, feed_dict=feed_dict)
        return errors

    def _plot_errors(self):
        """
        Plots and saves the error curves for all the tasks.

        return None
        """
        for task_id in self.task_ids:
            x = np.arange(len(self.training_errors[task_id]))
            fig, ax = plt.subplots(1, 1)
            plt.plot(x, self.training_errors[task_id], 'r', label='training')
            plt.plot(x, self.validation_errors[task_id], 'b', label='validation')
            plt.legend(loc="best", framealpha=0.3)
            fig.savefig("error-curve-task-{}.png".format(task_id))
        plt.close('all')


def main(args):
    """
    Runs the pipeline on the Million Songs Data set.
    :param args: Command-line arguments parsed with argparse.
    :return: None
    """

    # List of Labels to be used in the experiment.
    task_ids = [Labels.hotness.value, Labels.key.value, Labels.loudness.value]

    # Get the training, validation and testing set data and ground-truths
    x_train, x_validate, x_test, y_train, y_validate, y_test = fetch_data(task_ids)

    exp = Experiment(expt_name="some-meaningful-name", task_ids=task_ids, x_train=x_train, x_validate=x_validate,
                     x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                     model_class=LowLevelSharingModel, learning_rate=args.learning_rate, batch_size=args.batch_size,
                     num_epochs=args.num_epochs, checkpoint_freq=args.checkpoint_freq,
                     evaluation_freq=args.evaluation_freq)
    exp.initialize_network()
    exp.train()
    print("------")
    print("Training complete. Logs, outputs, and model saved in " + os.getcwd())


def dummy(args):
    """
    Runs the pipeline on a small synthetic dataset.
    :param args: Command line arguments parsed with argpase.
    :return: None
    """

    task_ids = ['1', '2', '3']
    input_dimension = 5000  # Dimensionality of each training set
    num_inputs_train = 750
    num_inputs_validate = 100
    num_inputs_test = 150

    # Training set
    x_train = np.random.random((num_inputs_train, input_dimension))
    y_train = {'1': np.random.random((num_inputs_train, 1)),
               '2': np.random.random((num_inputs_train, 1)),
               '3': np.random.random((num_inputs_train, 1))}

    # Validation set
    x_validate = np.random.random((num_inputs_validate, input_dimension))
    y_validate = {'1': np.random.random((num_inputs_validate, 1)),
                  '2': np.random.random((num_inputs_validate, 1)),
                  '3': np.random.random((num_inputs_validate, 1))}

    # Testing set
    x_test = np.random.random((num_inputs_test, input_dimension))
    y_test = {'1': np.random.random((num_inputs_test, 1)),
              '2': np.random.random((num_inputs_test, 1)),
              '3': np.random.random((num_inputs_test, 1))}

    exp = Experiment(expt_name="synthetic", task_ids=task_ids, x_train=x_train, x_validate=x_validate,
                     x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                     model_class=LowLevelSharingModel, learning_rate=args.learning_rate, batch_size=args.batch_size,
                     num_epochs=args.num_epochs, checkpoint_freq=args.checkpoint_freq,
                     evaluation_freq=args.evaluation_freq)
    exp.initialize_network()
    exp.train()
    print("Training complete. Logs, outputs, and model saved in " + os.getcwd())


if __name__ == '__main__':
    args = parse_arguments()
    print("Using the following values for hyperparameters:")
    print("Learning rate = {}".format(args.learning_rate))
    print("Mini-batch size = {}".format(args.batch_size))
    print("Number of epochs = {}".format(args.num_epochs))
    print("Checkpoint frequency= {}".format(args.checkpoint_freq))
    print("Evaluation frequency = {}".format(args.evaluation_freq))

    dummy(args)
