import tensorflow as tf

from activation import leaky_relu, relu
from loss import mse
from parameters import bias_variable, weight_variable
from regularization import batch_norm_layer, dropout_layer


class Layers(object):
    def __init__(self):
        """
        self.layers = a dictionary of all the layers keyed by the layer_id. Each layer_id should be unique
        """
        self.layers = dict()
        self.name = ''

    def name_network(self, name):
        self.name = name

    def network_type(self, is_first):
        self.is_first = is_first

    def get_layer(self, layer_id):
        """
        Retuns the tensorflow object corresponding to the requested layer
        :param layer_id: Layer identifier
        :return: Tensorflow layer object
        """
        assert not self._layer_verifier(layer_id), 'Invalid: Layer not present'
        return self.layers[layer_id]

    def add_input_layer(self, width, layer_name='input'):
        """
        Adds input layer to the model
        :param width: The width of the input = dimension of the input
        :param layer_name: The name of the layer. Type=string
        :return: None
        """
        assert self._layer_verifier(layer_name), 'Invalid: This layer is already present.'
        self.layers[layer_name] = tf.placeholder("float", [None, width])

        return layer_name

    def add_hidden_layer(self, input_layer_id, input_width, output_width,
                         layer_name, batch_norm=True, sharing=False):
        """
        Adds the hidden layer to the model
        :param input_layer_id: The input layer identifier
        :param input_width: The width of the input for this layer
        :param output_width: The width of the output for this layer
        :param layer_name: The name of the layer. Type=string
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        scope = self._get_scope(layer_name, layer_id, sharing)
        with tf.variable_scope(scope):
            assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'

            reuse = sharing and (not self.is_first)
            with tf.variable_scope("hello", reuse=reuse):
                    weights = weight_variable([input_width, output_width], "weight")
                    biases = bias_variable([output_width], "bias")
            if batch_norm:
                self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights)
            else:
                self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights) + biases

        # print weights
        # print biases

        return layer_id

    def add_regularization_layer(self, input_layer_id, layer_name, regularization_type='dropout',
                                 epsilon=None, dropout_ratio=None):
        """
        Adds the regularization layer to the model
        :param input_layer_id: The input layer identifier
        :param layer_name: The name of the layer. Type=string
        :param regularization_type: 'dropout' for Dropout and 'batch_norm' for Batch Normalization. Default = 'dropout'
        :param epsilon: The batch_norm parameter to ensure that division is not by zero when variance of batch = 0
        :param dropout_ratio: The fraction of the layers to be masked.
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'
        if regularization_type == 'dropout':
            if dropout_ratio:
                self.layers[layer_id] = dropout_layer(self.layers[input_layer_id], dropout_ratio)
            else:
                self.layers[layer_id] = dropout_layer(self.layers[input_layer_id])
        elif regularization_type == 'batch_norm':
            if epsilon:
                self.layers[layer_id] = batch_norm_layer(self.layers[input_layer_id], epsilon)
            else:
                self.layers[layer_id] = batch_norm_layer(self.layers[input_layer_id])
        else:
            raise ValueError('The type of regularization can only be one of ["dropout", "batch_norm"]')

        return layer_id

    def add_output_layer(self, input_layer_id, input_width, output_width, layer_name='output'):
        """
        Adds the output layer to the model
        :param input_layer_id: The input layer identifier
        :param input_width: The width of the input for this layer
        :param output_width: The width of the output for this layer
        ::param layer_name: The name of the layer. Type=string
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'
        weights = weight_variable([input_width, output_width])
        biases = bias_variable([output_width])
        self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights) + biases

        return layer_id

    def add_ground_truth_layer(self, width, layer_name='ground_truth'):
        """
        Adds ground truth layer to the model
        :param width: The width of the ground truth = dimension of the input
        :param layer_name: The name of the layer. Type=string
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'
        self.layers[layer_id] = tf.placeholder("float", [None, width])

        return layer_id

    def add_activation_layer(self, input_layer_id, layer_name, activation_type='relu'):
        """
        Adds the activation layer
        :param input_layer_id: The input layer identifier
        :param layer_name: The name of the layer. Type=string
        :param activation_type: 'relu' for RELU and 'leaky-relu' for Leaky RELU. Default = RELU
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'
        if activation_type == 'relu':
            self.layers[layer_id] = relu(self.layers[input_layer_id])
        elif activation_type == 'leaky-relu':
            self.layers[layer_id] = leaky_relu(self.layers[input_layer_id])
        else:
            raise ValueError('The type of activation can only be one of ["relu", "leaky-relu"]')

        return layer_id

    def add_loss_layer(self, layer_name, prediction_layer_id, ground_truth_layer_id, loss_type='mse'):
        """
        Adds a layer corresponding to the loss function
        :param layer_name: The name of the layer. Type=string
        :param prediction_layer_id: The identifier for the prediction layer
        :param ground_truth_layer_id: The identifier for the ground truth layer
        :param loss_type: 'mse' for MSE
        :return: None
        """
        layer_id = self._get_layer_id(layer_name)
        assert self._layer_verifier(layer_id), 'Invalid: This layer is already present.'
        assert not self._layer_verifier(prediction_layer_id), 'Invalid: Output layer id is invalid.'
        assert not self._layer_verifier(ground_truth_layer_id), 'Invalid: Ground truth layer id is invalid.'

        output = self.layers[prediction_layer_id]
        ground_truth = self.layers[ground_truth_layer_id]
        if loss_type == 'mse':
            self.layers[layer_id] = mse(ground_truth, output)
        else:
            raise ValueError('The type of loss can only be one of ["mse"]')

    def _get_scope(self, layer_name, layer_id, sharing):
        if sharing:
            return layer_name
        else:
            return layer_id

    def _get_layer_id(self, layer_name):
        return self.name + '-' + layer_name

    def _layer_verifier(self, layer_id):
        """
        Verifies if the layer asked to be formed is a valid layer and is not already formed before
        :param layer_id: The unique id of the layer. Type=string
        :return: True, if the layer is valid, else False
        """
        return layer_id not in self.layers
