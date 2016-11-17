import tensorflow as tf


class Optimizer(object):
    def __init__(self, cost):
        self.cost = cost

    def get_ada_delta(self, learning_rate):
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)

    def get_adam(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def get_adagrad(self, learning_rate):
        return tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)

    def get_momentum(self, learning_rate, momentum):
        return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)
