import tensorflow as tf


class Optimizer(object):
    def __init__(self, cost):
        """
        Optimizer class consisting of all possible optimizers that can be used
        :param cost: The cost placeholder that is to be optimized upon
        """
        self.cost = cost

    def get_ada_delta(self, learning_rate):
        """
        Ada Delta Optimizer
        :param learning_rate: The learning rate
        :return: the optimizer
        """
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)

    def get_adam(self, learning_rate):
        """
        Adam optimizer
        :param learning_rate: The learning rate
        :return: the optimizer
        """
        return tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def get_adagrad(self, learning_rate):
        """
        Adagrad optimizer
        :param learning_rate: The learning rate
        :return: the optimizer
        """
        return tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)

    def get_momentum(self, learning_rate, momentum):
        """
        Momentum optimizer
        :param learning_rate: The learning rate
        :param momentum: The momentum parameter for the optimizer
        :return: the optimizer
        """
        return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)
