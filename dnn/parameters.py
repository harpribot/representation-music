import tensorflow as tf


def weight_variable(shape):
    """
    This creates the weight placeholder for the DNN layers
    :param shape: shape of the weight variable
    :return: The tensorflow placeholder for weights
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    This creates the bias placeholder for the DNN layers
    :param shape: shape of the bias variable
    :return: The tensorflow placeholder for the bias
    """
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
