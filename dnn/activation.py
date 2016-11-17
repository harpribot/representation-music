import tensorflow as tf


def relu(features):
    return tf.nn.relu(features)


def leaky_relu(features, alpha=0.01):
    return tf.maximum(alpha * features, features)
