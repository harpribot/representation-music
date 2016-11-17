import tensorflow as tf
from parameters import weight_variable, bias_variable
from regularization import dropout_layer, batch_norm_layer
from activation import relu, leaky_relu


class Layers(object):
    def __init__(self):
        self.layers = dict()

    def _add_input_layer(self, width, layer_id='input'):
        self.layers[layer_id] = tf.placeholder("float", [None, width])

    def _add_hidden_layer(self, input_layer, input_width, output_width, layer_id):
        weights = weight_variable([input_width, output_width])
        biases = bias_variable([output_width])
        self.layers[layer_id] = tf.matmul(input_layer, weights) + biases

    def _add_regularization_layer(self, input_layer, layer_id, regulariztion_type='dropout',
                                  weights=None, epsilon=None, dropout_ratio=None):
        if regulariztion_type == 'dropout':
            if dropout_ratio:
                self.layers[layer_id] = dropout_layer(input_layer, dropout_ratio)
            else:
                self.layers[layer_id] = dropout_layer(input_layer)
        elif regulariztion_type=='batch_norm':
            assert weights is not None, 'Weights must be provided..'
            if epsilon:
                self.layers[layer_id] = batch_norm_layer(input_layer, weights, epsilon)
            else:
                self.layers[layer_id] = batch_norm_layer(input_layer, weights)
        else:
            raise ValueError('The type of regularization can only be one of ["dropout", "batch_norm"]')

    def _add_output_layer(self, input_layer, input_width, output_width, layer_id='output'):
        weights = weight_variable([input_width, output_width])
        biases = bias_variable([output_width])
        self.layers[layer_id] = tf.matmul(input_layer, weights) + biases

    def _add_activation_layer(self, input_layer, layer_id):
        self.layers[layer_id] = relu(input_layer)
