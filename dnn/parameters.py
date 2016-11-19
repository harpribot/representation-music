import tensorflow as tf


def weight_variable(shape, scope=None):
    """
    This creates the weight placeholder for the DNN layers
    :param shape: shape of the weight variable
    :return: The tensorflow placeholder for weights
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    if scope:
        return tf.get_variable(name=scope, dtype=tf.float32, initializer=initial)
    else:
        return tf.Variable(initial)


def bias_variable(shape, scope=None):
    """
    This creates the bias placeholder for the DNN layers
    :param shape: shape of the bias variable
    :return: The tensorflow placeholder for the bias
    """
    initial = tf.constant(0.01, shape=shape)
    if scope:
        return tf.get_variable(name=scope, dtype=tf.float32, initializer=initial)
    else:
        return tf.Variable(initial)
