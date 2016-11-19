import tensorflow as tf
from Models.models import LowLevelSharingModel
from dnn.optimizer import Optimizer
import numpy as np
from dnn.loss import mse

LEARNING_RATE = 1e-4


class Experiment():
    def __init__(self, task_ids, input_dimension, output_dimensions, input_data, labels):
        """
        Class to run experiments.
        :param task_ids: List of task identifiers
        :param input_dimension: Input dimension
        :param output_dimensions: Dictionary of output dimensions indexed by task identifiers
        :param input_data: Input set
        :param labels: Ground-truth labels for the input
        :return: None
        """
        self.sess = None
        self.task_ids = task_ids
        self.input_data = input_data
        self.labels = labels
        self.model = LowLevelSharingModel(task_ids, input_dimension, output_dimensions)

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self.model.create_model()
        self.__initialize_trainer()
        self.sess.run(tf.initialize_all_variables())

    def __initialize_trainer(self):
        self.cost = mse(0., 0.)
        for task_id in self.task_ids:
            self.cost += self.model.get_layer(task_id + '-loss')

        opt = Optimizer(self.cost)
        self.optimizer = opt.get_adagrad(LEARNING_RATE)

    def train(self):
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = [self.input_data]
        for task_id in self.task_ids:
            feed_dict[self.model.get_layer(task_id + '-ground-truth')] = [[self.labels[task_id]]]

        self.sess.run(self.optimizer, feed_dict=feed_dict)

        # see both of the outputs
        print self.sess.run([self.model.get_layer(tid + '-prediction') for tid in self.task_ids], feed_dict=feed_dict)


def main():
    task_ids = ['1', '2', '3']
    input_dimension = 5000
    input_data = np.random.rand(input_dimension)
    output_dimensions = {'1': 1, '2': 1, '3': 1}
    labels = {'1': 0.5, '2': 0.8, '3': 10.}
    exp = Experiment(task_ids, input_dimension, output_dimensions, input_data, labels)
    exp.initialize_network()
    exp.train()


if __name__ == '__main__':
    main()
