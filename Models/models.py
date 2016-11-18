import tensorflow as tf
from dnn.layers import Layers


class LowLevelSharingModel(Layers):
    def __init__(self, input_info, output_info):
        self.input_id, self.input_size = input_info
        self.output_info = output_info
        Layers.__init__(self)

    def _create_model(self):
        print 'Adding Input Layer'
        self._add_input_layer(width=self.input_size, layer_id=self.input_id)

        # First hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 1'
        self._add_hidden_layer(input_layer_id='input', input_width=self.input_size, output_width=1024,
                               layer_id='layer-1')
        self._add_activation_layer(input_layer_id='layer-1', layer_id='layer-1-relu')

        # Second hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 2'
        self._add_hidden_layer(input_layer_id='layer-1-relu', input_width=1024, output_width=512,
                               layer_id='layer-2')
        self._add_activation_layer(input_layer_id='layer-2', layer_id='layer-2-relu')

        # Third hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 3'
        self._add_hidden_layer(input_layer_id='layer-2-relu', input_width=512, output_width=256,
                               layer_id='layer-3')
        self._add_activation_layer(input_layer_id='layer-3', layer_id='layer-3-relu')
        self._add_regularization_layer(input_layer_id='layer-3-relu', layer_id='layer-3-dropout', dropout_ratio=0.5)

        # Fourth hidden layer for each task
        for output_id, _ in self.output_info:
            print 'Adding Hidden Layer 4 for Task-' + output_id
            layer_id = 'task-' + output_id + '-layer-4'
            self._add_hidden_layer(input_layer_id='layer-3-dropout', input_width=256, output_width=128,
                                   layer_id=layer_id)
            self._add_activation_layer(input_layer_id=layer_id, layer_id=layer_id + '-relu')
            self._add_regularization_layer(input_layer_id=layer_id + '-relu', layer_id=layer_id + '-dropout',
                                           dropout_ratio=0.5)

        # Output layer for each task
        for output_id, output_dim in self.output_info:
            print 'Adding Output Layer for Task-' + output_id
            self._add_output_layer(input_layer_id='task-' + output_id + '-layer-4-dropout', input_width=128,
                                   output_width=output_dim, layer_id='output-' + output_id)

        # Ground truth layer for each task
        for output_id, output_dim in self.output_info:
            print 'Adding Ground Truth Layer for Task-' + output_id
            self._add_ground_truth_layer(width=output_dim, layer_id='ground-truth-' + output_id)

        # Loss layer for each task
        for output_id, output_dim in self.output_info:
            print 'Adding Loss Layer for Task-' + output_id
            self._add_loss_layer(layer_id='loss-' + output_id, output_layer_id='output-' + output_id,
                                 ground_truth_layer_id='ground-truth-' + output_id, loss_type='mse')
