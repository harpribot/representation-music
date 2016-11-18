import tensorflow as tf
from Models.models import LowestSharingModel


class Experiment(LowestSharingModel):
    def __init__(self, input_info, output_info):
        LowestSharingModel.__init__(self, input_info, output_info)
        self.sess = None

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self._create_model()


