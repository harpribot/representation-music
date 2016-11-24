import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dnn.loss import mse
from dnn.optimizer import Optimizer
from Models.high_level_sharing_model import HighLevelSharingModel
from Models.interspersed_sharing_model import InterspersedSharingModel
from Models.low_level_sharing_model import LowLevelSharingModel
from utils.msd import MillionSongDataset
from utils.params import BATCH_SIZE, NUM_EPOCHS, FREQ_OF_CHECKPOINTS, FREQ_OF_EVALUATIONS, LEARNING_RATE
from utils.params import EXPT_DIRECTORY_PATH, FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS
from utils.params import TRAIN_FRACTION, TEST_FRACTION, VALIDATE_FRACTION, TOTAL_NUM_EXAMPLES
from utils.params import Task


class Experiment():
    def __init__(self, task_ids,
                 x_train, x_validate, x_test, y_train, y_validate, y_test, model_class, expt_name):
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
        :return: None
        """
        self.task_ids = task_ids
        self.x_train = x_train
        self.x_validate = x_validate
        self.x_test = x_test
        self.y_train = y_train
        self.y_validate = y_validate
        self.y_test = y_test

        self.sess = None
        self.optimizer = None
        self.saver = None

        input_dimension = self.x_train.shape[1]
        output_dimensions = {}
        for task_id in task_ids:
            output_dimensions[task_id] = self.y_train[task_id].shape[1]

        self.model = model_class(task_ids, input_dimension, output_dimensions)

        # Dictionary of list of training errors indexed by task-identifiers. Each list contains errors of the model
        # on the training set for that task over the course of training.
        self.training_errors = {}

        # Dictionary of list of validation errors indexed by task-identifiers. Each list contains errors of the model
        # on the validation set for that task over the course of training.
        self.validation_errors = {}

        self._initialize_error_dictionaries()
        self._create_expt_directory(expt_name)

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self.model.create_model()
        self._initialize_trainer()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def train(self):
        print("Starting training")
        step = 0
        start_time = time.time()
        for epoch in xrange(1, NUM_EPOCHS + 1):
            if epoch == NUM_EPOCHS:
                print("Last epoch. All minibatches will be evaluated and checkpointed.")
            for minibatch_indices in self._iterate_minibatches(BATCH_SIZE):
                feed_dict = dict()
                feed_dict[self.model.get_layer('input')] = self.x_train[minibatch_indices]
                for id_ in self.task_ids:
                    feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_train[id_][minibatch_indices]

                self.optimizer.run(session=self.sess, feed_dict=feed_dict)

                step += 1
                duration = int(time.time() - start_time)

                # Evaluate model fairly often, including on the last epoch.
                if step % FREQ_OF_EVALUATIONS == 0 or epoch == NUM_EPOCHS:
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
                if step % FREQ_OF_CHECKPOINTS == 0 or epoch == NUM_EPOCHS:
                    self.saver.save(self.sess, 'checkpoint-' + str(step).zfill(8))

    def _initialize_error_dictionaries(self):
        for task_id in self.task_ids:
            self.training_errors[task_id] = []
            self.validation_errors[task_id] = []

    def _create_expt_directory(self, expt_name):
        """
        Creates a directory for this experiment where all the logs and outputs are saved. The directory is named
        after the experiment-name suffixed with a 4-digit random number. The 
        :param expt_name: Experiment name.
        :return: None
        """
        os.chdir(EXPT_DIRECTORY_PATH)   # Change working directory to the Experiments folder
        expt_dir = expt_name + "-" + str(np.random.randint(1000, 9999))
        try:
            os.makedirs(expt_dir)
        except OSError:
            if not os.path.isdir(expt_dir):
                raise
        finally:
            os.chdir("./" + expt_dir)   # Change working directory to the newly created folder
            print("Experiment Directory: " + expt_dir)

    def _initialize_trainer(self):
        self.cost = mse(0., 0.)
        for task_id in self.task_ids:
            self.cost += self.model.get_layer(task_id + '-loss')

        opt = Optimizer(self.cost)
        self.optimizer = opt.get_adagrad(LEARNING_RATE)

    def _iterate_minibatches(self, batchsize, shuffle=True):
        """Yields list of indices for each minibatch. The last minibatch is not used because it usually has 
        a different size. Since training set is shuffled before creating minibatches, the training examples in the
        ignored minibatches are used in some other minibatch in at least one epoch with high probability.
        :param batchsize: Size of each minibatch
        :param shuffle: Boolean to be set to True if the training set needs to be shuffled at each epoch.
        :yield: List of indices corresponding to a minibatch
        """
        num_input = self.x_train.shape[0]
        indices = np.arange(num_input)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_input - batchsize + 1, batchsize):
            minibatch = indices[start_idx:start_idx + batchsize]
            yield minibatch

    def _training_errors(self):
        """Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the training set."""
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_train
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_train[id_]
        errors = {}
        for task_id in self.task_ids:
            errors[task_id] = self.model.get_layer(task_id + '-loss').eval(session=self.sess, feed_dict=feed_dict)
        return errors

    def _validation_errors(self):
        """Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the training set."""
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_validate
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_validate[id_]
        errors = {}
        for task_id in self.task_ids:
            errors[task_id] = self.model.get_layer(task_id + '-loss').eval(session=self.sess, feed_dict=feed_dict)
        return errors

    def _plot_errors(self):
        """Plots and saves the error curves for all the tasks."""
        for task_id in self.task_ids:
            x = np.arange(len(self.training_errors))
            fig, ax = plt.subplots(1, 1)
            plt.plot(x, self.training_errors[task_id], 'r', label='training')
            plt.plot(x, self.validation_errors[task_id], 'b', label='validation')
            plt.legend(loc="best", framealpha=0.3)
            fig.savefig("error-curve-task-{}.png".format(task_id))


def main():
    """Runs the pipeline on the Million Songs Dataset."""

    # List of Tasks to be used in the experiment.
    task_ids = [Task.hotttnesss.value, Task.key.value, Task.loudness.value]

    # Load the dataset in memory.
    db = MillionSongDataset(FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS)
    # Generate train/validation/test splits.
    db.generate_split(TRAIN_FRACTION, VALIDATE_FRACTION, TEST_FRACTION, TOTAL_NUM_EXAMPLES)

    # Training set
    print("Creating training set")
    x_train = np.array([bow for bow in db.get_bow(db.train)], dtype=float)
    labels_train = np.array([t.vector() for t in db.get_features(db.train)])

    # Validation set
    print("Creating validation set")
    x_validate = np.array([bow for bow in db.get_bow(db.validate)], dtype=float)
    labels_validate = np.array([t.vector() for t in db.get_features(db.validate)])

    # Testing set
    print("Creating testing set")
    x_test = np.array([bow for bow in db.get_bow(db.test)], dtype=float)
    labels_test = np.array([t.vector() for t in db.get_features(db.test)])

    # Build the label dictionary using tasks under consideration, discaring other labels.
    y_train = {}
    y_validate = {}
    y_test = {}
    for task_id in task_ids:
        y_train[task_id] = np.array(labels_train[:,int(task_id)], dtype=float).reshape(-1, 1)
        y_validate[task_id] = np.array(labels_validate[:,int(task_id)], dtype=float).reshape(-1, 1)
        y_test[task_id] = np.array(labels_test[:,int(task_id)], dtype=float).reshape(-1, 1)

    exp = Experiment(expt_name="some-meaningful-name", task_ids=task_ids, x_train=x_train, x_validate=x_validate,
                     x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                     model_class=LowLevelSharingModel)
    exp.initialize_network()
    exp.train()
    print("Training complete. Logs, outputs, and model saved in " + os.getcwd())


def dummy():
    """Runs the pipeline on a small synthetic dataset."""

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
                     model_class=LowLevelSharingModel)
    exp.initialize_network()
    exp.train()
    print("Training complete. Logs, outputs, and model saved in " + os.getcwd())


if __name__ == '__main__':
    dummy()
