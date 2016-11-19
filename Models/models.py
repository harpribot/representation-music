import tensorflow as tf
from dnn.layers import Layers


class LowLevelSharingModel(Layers):
    def __init__(self, input_info, output_info):
        """
        A low level sharing model
        :param input_info: (input_id , input_size)
        :param output_info: (output_id, output_size)
        """
        self.input_id, self.input_size = input_info
        self.output_info = output_info
        Layers.__init__(self)

    def _create_model(self):
        """
        Creates the model consisting of several parallel networks
        :return: None
        """
        self.__create_parallel_networks()

    def __create_parallel_networks(self):
        """
        Create all the parallel networks one by one
        :return: None
        """
        print 'Adding Input Layer'
        id_inp = self._add_input_layer(width=self.input_size, layer_name=self.input_id)

        is_first = True
        for output_id, output_dim in self.output_info:
            self.__create_network(output_id, output_dim, id_inp, is_first)
            is_first = False

    def __create_network(self, output_id, output_dim, input_id, is_first):
        """
        Create each network
        :param output_id: The id of the output labelling task
        :param output_dim: The prediction dimension
        :param input_id: The id of the input used for the network
        :return: None
        """
        self._name_network(output_id)
        self._network_type(is_first)

        # First hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 1 for Task-' + output_id
        id_hidden1 = self._add_hidden_layer(input_layer_id=input_id, input_width=self.input_size,
                                            output_width=1024, layer_name='layer-1', sharing=True)
        id_act1 = self._add_activation_layer(input_layer_id=id_hidden1, layer_name='layer-1-relu')

        # Second hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 2 for Task-' + output_id
        id_hidden2 = self._add_hidden_layer(input_layer_id=id_act1, input_width=1024,
                                            output_width=512, layer_name='layer-2')
        id_act2 = self._add_activation_layer(input_layer_id=id_hidden2, layer_name='layer-2-relu')

        # Third hidden layer which is shared across all tasks
        print 'Adding Hidden Layer 3 for Task-' + output_id
        id_hidden3 = self._add_hidden_layer(input_layer_id=id_act2, input_width=512,
                                            output_width=256, layer_name='layer-3')
        id_act3 = self._add_activation_layer(input_layer_id=id_hidden3, layer_name='layer-3-relu')
        id_reg3 = self._add_regularization_layer(input_layer_id=id_act3, layer_name='layer-3-dropout',
                                                 dropout_ratio=0.5)

        # Fourth hidden layer is not shared
        print 'Adding Hidden Layer 4 for Task-' + output_id
        id_hidden4 = self._add_hidden_layer(input_layer_id=id_reg3, input_width=256,
                                            output_width=128, layer_name='layer-4')
        id_act4 = self._add_activation_layer(input_layer_id=id_hidden4, layer_name='layer-4-relu')
        id_reg4 = self._add_regularization_layer(input_layer_id=id_act4, layer_name='layer-4-dropout',
                                                 dropout_ratio=0.5)

        # Output layer is not shared
        print 'Adding Output Layer for Task-' + output_id
        prediction_id = self._add_output_layer(input_layer_id=id_reg4, input_width=128,
                                               output_width=output_dim, layer_name='prediction')

        # Ground truth layer for the task
        print 'Adding Ground Truth Layer for Task-' + output_id
        groundtruth_id = self._add_ground_truth_layer(width=output_dim, layer_name='ground-truth')

        # Loss layer for the task
        print 'Adding Loss Layer for Task-' + output_id
        self._add_loss_layer(layer_name='loss', prediction_layer_id=prediction_id,
                             ground_truth_layer_id=groundtruth_id, loss_type='mse')
