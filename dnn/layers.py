import tensorflow as tf
from parameters import weight_variable, bias_variable
from regularization import dropout_layer, batch_norm_layer
from activation import relu, leaky_relu
from loss import mse


class Layers(object):
    def __init__(self):
        """
        self.layers = a dictionary of all the layers keyed by the layer_id. Each layer_id should be unique
        """
        self.layers = dict()

    def _add_input_layer(self, width, layer_id='input'):
        """
        Adds input layer to the model
        :param width: The width of the input = dimension of the input
        :param layer_id: The unique id of the layer. Type=string
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        self.layers[layer_id] = tf.placeholder("float", [None, width])

    def _add_hidden_layer(self, input_layer_id, input_width, output_width, layer_id, batch_norm=True):
        """
        Adds the hidden layer to the model
        :param input_layer_id: The input layer identifier
        :param input_width: The width of the input for this layer
        :param output_width: The width of the output for this layer
        :param layer_id: The unique id of the layer. Type=string
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        weights = weight_variable([input_width, output_width])
        biases = bias_variable([output_width])
        if batch_norm:
            self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights)
        else:
            self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights) + biases

    def _add_regularization_layer(self, input_layer_id, layer_id, regularization_type='dropout',
                                  weights=None, epsilon=None, dropout_ratio=None):
        """
        Adds the regularization layer to the model
        :param input_layer_id: The input layer identifier
        :param layer_id: The unique id of the layer. Type=string
        :param regularization_type: 'dropout' for Dropout and 'batch_norm' for Batch Normalization. Default = 'dropout'
        :param weights: The weights of the current layer
        :param epsilon: The batch_norm parameter to ensure that division is not by zero when variance of batch = 0
        :param dropout_ratio: The fraction of the layers to be masked.
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        if regularization_type == 'dropout':
            if dropout_ratio:
                self.layers[layer_id] = dropout_layer(self.layers[input_layer_id], dropout_ratio)
            else:
                self.layers[layer_id] = dropout_layer(self.layers[input_layer_id])
        elif regularization_type == 'batch_norm':
            assert weights is not None, 'Weights must be provided..'
            if epsilon:
                self.layers[layer_id] = batch_norm_layer(self.layers[input_layer_id], weights, epsilon)
            else:
                self.layers[layer_id] = batch_norm_layer(self.layers[input_layer_id], weights)
        else:
            raise ValueError('The type of regularization can only be one of ["dropout", "batch_norm"]')

    def _add_output_layer(self, input_layer_id, input_width, output_width, layer_id='output'):
        """
        Adds the output layer to the model
        :param input_layer_id: The input layer identifier
        :param input_width: The width of the input for this layer
        :param output_width: The width of the output for this layer
        :param layer_id: The unique id of the layer. Type=string
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        weights = weight_variable([input_width, output_width])
        biases = bias_variable([output_width])
        self.layers[layer_id] = tf.matmul(self.layers[input_layer_id], weights) + biases

    def _add_ground_truth_layer(self, width, layer_id='ground_truth'):
        """
        Adds ground truth layer to the model
        :param width: The width of the ground truth = dimension of the input
        :param layer_id: The unique id of the layer. Type=string
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        self.layers[layer_id] = tf.placeholder("float", [None, width])

    def _add_activation_layer(self, input_layer_id, layer_id, activation_type='relu'):
        """
        Adds the activation layer
        :param input_layer_id: The input layer identifier
        :param layer_id: The unique id of the layer. Type=string
        :param activation_type: 'relu' for RELU and 'leaky-relu' for Leaky RELU. Default = RELU
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        if activation_type == 'relu':
            self.layers[layer_id] = relu(self.layers[input_layer_id])
        elif activation_type == 'leaky-relu':
            self.layers[layer_id] = leaky_relu(self.layers[input_layer_id])
        else:
            raise ValueError('The type of activation can only be one of ["relu", "leaky-relu"]')

    def __layer_verifier(self, layer_id):
        """
        Verifies if the layer asked to be formed is a valid layer and is not already formed before
        :param layer_id: The unique id of the layer. Type=string
        :return: True, if the layer is valid, else False
        """
        return layer_id not in self.layers

    def _add_loss_layer(self, layer_id, output_layer_id, ground_truth_layer_id, loss_type='mse'):
        """
        Adds a layer corresponding to the loss function
        :param layer_id: The loss layer identifier
        :param output_layer_id: The output layer identifier
        :param ground_truth_layer_id: The ground truth layer identifier
        :param loss_type: 'mse' for MSE
        :return: None
        """
        assert self.__layer_verifier(layer_id), 'Invalid: This layer is already present.'
        assert not self.__layer_verifier(output_layer_id), 'Invalid: Output layer id is invalid.'
        assert not self.__layer_verifier(ground_truth_layer_id), 'Invalid: Ground truth layer id is invalid.'

        output = self.layers[output_layer_id]
        ground_truth = self.layers[ground_truth_layer_id]
        if loss_type == 'mse':
            self.layers[layer_id] = mse(ground_truth, output)
        else:
            raise ValueError('The type of loss can only be one of ["mse"]')
