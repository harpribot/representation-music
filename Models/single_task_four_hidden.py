import sys

from model import Model


class SingleTaskFourHiddenModel(Model):
    def __init__(self, task_ids, input_dimension, output_dimensions):
        """
        A single task model with four hidden layers.
        :param task_ids: The task ask identifier as a a single-element list
        :param input_dimension: Input dimension
        :param output_dimensions: Dictionary of output dimensions indexed by the task identifier.
        """
        assert len(task_ids) == 1, "Multiple tasks specified for a single-task model."

        Model.__init__(self, task_ids, input_dimension, output_dimensions)

    def create_model(self):
        """
        Creates the model consisting of a single network.
        :return: None
        :return: None
        """
        # print 'Adding Input Layer'
        input_layer_id = self.add_input_layer(width=self.input_dimension, layer_name=self.input_id)

        is_first = True
        task_id = self.task_ids[0]
        sys.stderr.write('Creating network for Task-' + task_id + '\n')
        self._create_network(task_id, input_layer_id, is_first)
        # is_first = False

    def _create_network(self, task_id, input_layer_id, is_first):
        """
        Create the network
        :param task_id: Task identifier
        :param input_layer_id: The id of the input used for the network
        :param is_first: Boolean which is True if the network is the first one being created.
        :return: None
        """
        self.name_network(task_id)
        self.network_type(is_first)

        # First hidden layer;
        # print 'Adding Hidden Layer 1 for Task-' + task_id
        id_hidden1 = self.add_hidden_layer(input_layer_id=input_layer_id,
                                           input_width=self.input_dimension,
                                           output_width=1024,
                                           layer_name='layer-1')

        id_act1 = self.add_activation_layer(input_layer_id=id_hidden1,
                                            layer_name='layer-1-relu')

        # Second hidden layer;
        # print 'Adding Hidden Layer 2 for Task-' + task_id
        id_hidden2 = self.add_hidden_layer(input_layer_id=id_act1,
                                           input_width=1024,
                                           output_width=512,
                                           layer_name='layer-2')

        id_act2 = self.add_activation_layer(input_layer_id=id_hidden2,
                                            layer_name='layer-2-relu')

        # Third hidden layer;
        # print 'Adding Hidden Layer 3 for Task-' + task_id
        id_hidden3 = self.add_hidden_layer(input_layer_id=id_act2,
                                           input_width=512,
                                           output_width=256,
                                           layer_name='layer-3')

        id_act3 = self.add_activation_layer(input_layer_id=id_hidden3,
                                            layer_name='layer-3-relu')

        id_reg3 = self.add_regularization_layer(input_layer_id=id_act3,
                                                layer_name='layer-3-dropout',
                                                dropout_ratio=0.5)

        # Fourth hidden layer
        # print 'Adding Hidden Layer 4 for Task-' + task_id
        id_hidden4 = self.add_hidden_layer(input_layer_id=id_reg3,
                                           input_width=256,
                                           output_width=128,
                                           layer_name='layer-4')

        id_act4 = self.add_activation_layer(input_layer_id=id_hidden4,
                                            layer_name='layer-4-relu')

        id_reg4 = self.add_regularization_layer(input_layer_id=id_act4,
                                                layer_name='layer-4-dropout',
                                                dropout_ratio=0.5)

        # Output layer
        # print 'Adding Output Layer for Task-' + task_id
        prediction_id = self.add_output_layer(input_layer_id=id_reg4,
                                              input_width=128,
                                              output_width=self.output_dimensions[task_id],
                                              layer_name='prediction')

        # Ground truth layer for the task
        # print 'Adding Ground Truth Layer for Task-' + task_id
        groundtruth_id = self.add_ground_truth_layer(width=self.output_dimensions[task_id],
                                                     layer_name='ground-truth')

        # Loss layer for the task
        # print 'Adding Loss Layer for Task-' + task_id
        self.add_loss_layer(layer_name='loss',
                            prediction_layer_id=prediction_id,
                            ground_truth_layer_id=groundtruth_id,
                            loss_type='mse')
