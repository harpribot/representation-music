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


def cross_entropy(ground_truth, output):
    """
    Returns the Cross-Entropy loss container
    
    :param ground_truth: The ground truth layer
    :param output: The network output layer
    :return: Cross-entropy error
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, ground_truth))
