from abc import abstractmethod

from dnn.layers import Layers


class Model(Layers):
    def __init__(self, task_ids, input_dimension, output_dimensions):
        """
        The model class
        :param task_ids: List of task identifiers
        :param input_dimension: Input dimension
        :param output_dimensions: Dictionary of output dimensions indexed by task identifiers
        """
        self.task_ids = task_ids
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_id = 'input'

        Layers.__init__(self)

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def __create_network(self, task_id, input_layer_id, is_first):
        pass
