import tensorflow as tf


def mse(ground_truth, output):
    """
    Returns the Mean Squared Error loss container
    :param ground_truth: The ground truth layer
    :param output: The network output layer
    :return: MSE error
    """
    # return tf.nn.l2_loss(ground_truth - output)
    return tf.reduce_mean((ground_truth - output)**2)
