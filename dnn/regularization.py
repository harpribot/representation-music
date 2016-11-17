import tensorflow as tf


def dropout_layer(input_layer, dropout_ratio=0.5):
    return tf.nn.dropout(input_layer, dropout_ratio)


def batch_norm_layer(input_layer, weights, epsilon):
    z = tf.matmul(input_layer, weights)
    output_dim = weights.get_shape()[2].value
    batch_mean, batch_var = tf.nn.moments(z, [0])
    scale = tf.Variable(tf.ones([output_dim]))
    beta = tf.Variable(tf.ones([output_dim]))
    return tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, epsilon)
