import tensorflow as tf
from Models.models import LowLevelSharingModel
from dnn.optimizer import Optimizer
import numpy as np
from dnn.loss import mse

LEARNING_RATE = 1e-4


class Experiment(LowLevelSharingModel):
    def __init__(self, input_info, output_info, input_data, output_labels):
        LowLevelSharingModel.__init__(self, input_info, output_info)
        self.sess = None
        self.input_data = input_data
        self.output_labels = output_labels

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self._create_model()
        self.__initialize_trainer()
        self.sess.run(tf.initialize_all_variables())

    def __initialize_trainer(self):
        self.cost = mse(0., 0.)
        for output_id, _ in self.output_info:
            self.cost += self.layers['loss-' + output_id]

        opt = Optimizer(self.cost)
        self.optimizer = opt.get_adagrad(LEARNING_RATE)

    def train(self):
        feed_dict = dict()
        feed_dict[self.layers['input']] = [self.input_data]
        for output_id, _ in self.output_info:
            feed_dict[self.layers['ground-truth-' + output_id]] = [[self.output_labels[output_id]]]

        self.sess.run(self.optimizer, feed_dict=feed_dict)

        # see both of the outputs
        print self.sess.run([self.layers['output-' + '1'], self.layers['output-' + '2']], feed_dict=feed_dict)


input_data = np.random.rand(5000)
labels = {'1': 0.5, '2': 0.8}
input_info = ('input', 5000)
output_info = [('1', 1), ('2', 1)]
exp = Experiment(input_info, output_info, input_data, labels)
exp.initialize_network()
exp.train()
