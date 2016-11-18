import tensorflow as tf
from dnn.layers import Layers


class LowestSharingModel(Layers):
    def __init__(self, input_info, output_info):
        self.input_id, self.input_size = input_info
        self.output_info = output_info
        Layers.__init__()

    def _create_model(self):
        self._add_input_layer(width=self.input_size, layer_id=self.input_id)
        # First hidden layer
        self._add_hidden_layer(input_layer_id='input',input_width=self.input_size, output_width=1024,
                               layer_id='layer-1')
        self._add_activation_layer(input_layer_id='layer-1',layer_id='layer-1-relu')

        # Second hidden layer
        self._add_hidden_layer(input_layer_id='layer-1-relu', input_width=1024, output_width=512,
                               layer_id='layer-2')
        self._add_activation_layer(input_layer_id='layer-2', layer_id='layer-2-relu')

        # Third hidden layer
        self._add_hidden_layer(input_layer_id='layer-2-relu', input_width=512, output_width=256,
                               layer_id='layer-3')
        self._add_activation_layer(input_layer_id='layer-3', layer_id='layer-3-relu')
        self._add_regularization_layer(input_layer_id='layer-3-relu', layer_id='layer-3-dropout', dropout_ratio=0.5)

        # Fourth hidden layer
        self._add_hidden_layer(input_layer_id='layer-3-relu', input_width=256, output_width=128,
                               layer_id='layer-4')
        self._add_activation_layer(input_layer_id='layer-4', layer_id='layer-4-relu')
        self._add_regularization_layer(input_layer_id='layer-4-relu', layer_id='layer-4-dropout', dropout_ratio=0.5)

        # Output layer
        for output_id, output_dim in self.output_info:
            self._add_output_layer(input_layer_id='layer-4-dropout', input_width=128,
                                   output_width=output_dim, layer_id=output_id)


