import tensorflow as tf


def relu(features):
    """
    RELU activations
    :param features: The input to the RELU layer
    :return: relu activation output
    """
    return tf.nn.relu(features)


def leaky_relu(features, alpha=0.01):
    """
    Leaky RELU - Prevents dying neurons due to gradient explosion
    :param features: The input to the RELU layer
    :param alpha: The slope of the negative input values
    :return: relu activation output
    """
    return tf.maximum(alpha * features, features)
