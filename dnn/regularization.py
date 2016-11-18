import tensorflow as tf


def dropout_layer(input_layer, dropout_ratio=0.5):
    """
    Forms a dropout layer
    :param input_layer: The layer on which the dropout is to be applied
    :param dropout_ratio: The ratio of the dropout that is to be applied
    :return: The dropout layer
    """
    return tf.nn.dropout(input_layer, dropout_ratio)


def batch_norm_layer(input_layer, weights, epsilon=1e-5):
    """
    The batch normalization applied to the output of the layer before applying the activation.
    Note: This layer is internal to the hidden layer,
            and should be applied to the output of the hidden layer before applying the activation.
            The input_layer is the input the present hidden layer, the weights are the weights of the hidden layer.
    :param input_layer: The input of the present layer, and the output of the previous layer
    :param weights: The weights of the present layer
    :param epsilon: The small value added to the denominator to prevent division by 0, when the variance = 0
    :return: The batch normalization output. This output should be applied with the activation
    """
    z = tf.matmul(input_layer, weights)
    output_dim = weights.get_shape()[2].value
    batch_mean, batch_var = tf.nn.moments(z, [0])
    scale = tf.Variable(tf.ones([output_dim]))
    beta = tf.Variable(tf.ones([output_dim]))
    return tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, epsilon)
