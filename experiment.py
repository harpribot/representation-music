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
from utils.params import BATCH_SIZE, EXPT_DIRECTORY_PATH, FREQ_OF_CHECKPOINTS, LEARNING_RATE, NUM_EPOCHS


class Experiment():
    def __init__(self, task_ids, input_dimension, output_dimensions, inputs, labels, model_class, expt_name):
        """
        Class to run experiments.
        :param task_ids: List of task identifiers
        :param input_dimension: Input dimension
        :param output_dimensions: Dictionary of output dimensions indexed by task identifiers
        :param inputs: Input set
        :param labels: Ground-truth labels for the input
        :param model_class: A class derived from the Model class
        :param expt_name: Name for the experiment. This will be used as a prefix for the directory created to store
            the logs and output of this experiment.
        :return: None
        """
        self.task_ids = task_ids
        self.inputs = inputs
        self.labels = labels

        self.sess = None
        self.optimizer = None
        self.saver = None
        self.model = model_class(task_ids, input_dimension, output_dimensions)
        self.expt_dir = self._create_expt_directory(expt_name)

        self.training_errors = []
        self.validation_errors = []

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self.model.create_model()
        self.__initialize_trainer()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def train(self):
        step = 0
        start_time = time.time()
        for epoch in xrange(NUM_EPOCHS):

            for minibatch_indices in self._iterate_minibatches(BATCH_SIZE):
                feed_dict = dict()
                feed_dict[self.model.get_layer('input')] = self.inputs[minibatch_indices]
                for id_ in self.task_ids:
                    feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.labels[id_][minibatch_indices]

                self.optimizer.run(session=self.sess, feed_dict=feed_dict)

                step += 1
                duration = int(time.time() - start_time)

                # Evaluate and checkpoint every 100 steps
                if step % FREQ_OF_CHECKPOINTS == 0:
                    # Print current errors on training and validation sets.
                    t_errors = self._training_errors()
                    v_errors = self._validation_errors()
                    print("Step: {}, Epoch: {}, Duration: {}, Training Errors: {}, Validation Errors: {}"
                          .format(step, epoch, duration, t_errors, v_errors))

                    # Add current errors to the cummulative errors list for plotting.
                    self.training_errors.extend(t_errors)
                    self.validation_errors.extend(v_errors)
                    # self._plot_errors()

                    # Checkpoint model
                    self.saver.save(self.sess, 'checkpoint-' + str(step).zfill(8))

    def _create_expt_directory(self, expt_name):
        os.chdir(EXPT_DIRECTORY_PATH)   # Change working directory to the Experiments folder
        expt_dir = expt_name + "-" + str(np.random.randint(1000, 9999))
        try:
            os.makedirs(expt_dir)
        except OSError:
            if not os.path.isdir(expt_dir):
                raise
        finally:
            os.chdir(EXPT_DIRECTORY_PATH + "/" + expt_dir)   # Change working directory to the newly created folder
            print("Experiment Directory: " + expt_dir)
            return expt_dir

    def __initialize_trainer(self):
        self.cost = mse(0., 0.)
        for task_id in self.task_ids:
            self.cost += self.model.get_layer(task_id + '-loss')

        opt = Optimizer(self.cost)
        self.optimizer = opt.get_adagrad(LEARNING_RATE)

    def _iterate_minibatches(self, batchsize, shuffle=True):
        num_input = self.inputs.shape[0]
        indices = np.arange(num_input)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_input - batchsize + 1, batchsize):
            minibatch = indices[start_idx:start_idx + batchsize]
            yield minibatch

    def _training_errors(self):
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.inputs
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.labels[id_]
        errors = []
        for task_id in self.task_ids:
            errors.append(self.model.get_layer(task_id + '-loss')
                          .eval(session=self.sess, feed_dict=feed_dict))
        return errors

    def _validation_errors(self):
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.inputs
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.labels[id_]
        errors = []
        for task_id in self.task_ids:
            errors.append(self.model.get_layer(task_id + '-loss')
                          .eval(session=self.sess, feed_dict=feed_dict))
        return errors

    def _plot_errors(self):
        x = np.arange(len(self.training_errors))
        fig, ax = plt.subplots(1, 1)
        plt.plot(x, self.training_errors, 'r', label='training')
        plt.plot(x, self.validation_errors, 'b', label='validation')
        plt.legend(loc="best", framealpha=0.3)
        fig.savefig("error-curve.png")


def main():
    input_dimension = 5000
    num_inputs = 1000
    task_ids = ['1', '2', '3']
    output_dimensions = {'1': 1, '2': 1, '3': 1}

    inputs = np.random.random((num_inputs, input_dimension))
    labels = {'1': np.random.random((num_inputs, 1)),
              '2': np.random.random((num_inputs, 1)),
              '3': np.random.random((num_inputs, 1))}
    exp = Experiment(expt_name="dummy", task_ids=task_ids, input_dimension=input_dimension,
                     output_dimensions=output_dimensions, inputs=inputs, labels=labels,
                     model_class=LowLevelSharingModel)
    exp.initialize_network()
    exp.train()


if __name__ == '__main__':
    main()
