import tensorflow as tf
from Models.models import LowestSharingModel
import


class Experiment(LowestSharingModel):
    def __init__(self, input_info, output_info):
        LowestSharingModel.__init__(self, input_info, output_info)
        self.sess = None

    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        self._create_model()


input =
input_info = ('input', 5000)
output_info = [('output-1', 1), ('output-2', 1)]
exp = Experiment(input_info, output_info)
exp.initialize_network()


