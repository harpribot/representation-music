# import os
import numpy as np
import sys
import tensorflow as tf
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Models.low_level_sharing_four_hidden import LowLevelSharingModel
from utils.data_utils.labels import Labels
from utils.data_utils.data_handler import fetch_data
from utils.network_utils.params import LossTypes
from utils.data_utils.data_handler import convert_to_one_hot


class EvaluateModel(object):
    def __init__(self, task_ids):
        self.x_test = None
        self.y_test = {}
        self.input_dimension = 0
        self.output_dimensions = {}
        self.task_ids = task_ids
        self.model = None
        self.sess = None

    def load_model(self, model_file, model_class):
        """
        Loads the model from the given checkpoint file.

        :param model_file: The checkpoint file from which the model should be loaded.
        :param model_class: The :class:`Model` class or any of its child classes.
        """
        sys.stderr.write("------\n")
        self.sess = tf.Session()
        self.model = model_class(self.task_ids, self.input_dimension, self.output_dimensions)
        self.model.create_model()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        sys.stderr.write("------\n")
        sys.stderr.write("Model " + model_file + " loaded.\n")

    def load_data(self):
        """
        Loads the test dataset.
        """
        sys.stderr.write("Loading data.\n")
        _, _, self.x_test, _, _, self.y_test = fetch_data(self.task_ids)
        self.input_dimension = self.x_test.shape[1]
        self.output_dimensions = {task_id: self.y_test[task_id].shape[1] for task_id in self.task_ids.keys()}
        sys.stderr.write("Test set created.\n")


    def load_dummy_data(self):
        """
        Loads the dummy test dataset.
        """
        sys.stderr.write("Loading data.\n")
        task_ids = {'1': LossTypes.mse, '2': LossTypes.mse, '3': LossTypes.cross_entropy}
        self.input_dimension = 5000  # Dimensionality of each training set
        num_inputs_test = 150

        # Testing set
        self.x_test = np.random.random((num_inputs_test, self.input_dimension))

        for task_id, loss_type in task_ids.iteritems():
            if loss_type is LossTypes.mse:
                self.y_test[task_id] = np.random.random((num_inputs_test, 1))
            elif loss_type is LossTypes.cross_entropy:
                # Testing labels -- 2-dimensional one-hot vectors for each example.
                labels = np.random.binomial(1, 0.8, num_inputs_test).reshape(1, num_inputs_test)
                self.y_test[task_id] = convert_to_one_hot(labels)
        self.output_dimensions = {task_id: self.y_test[task_id].shape[1] for task_id in self.task_ids.keys()}
        sys.stderr.write("Test set created.\n")

    def evaluate_model(self):
        """
        Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the test set.

        :return dictionary of test errors
        """
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_test
        for id_ in self.task_ids.keys():
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_test[id_]
        errors = {}
        for task_id, loss_type in self.task_ids.iteritems():
            if loss_type is LossTypes.mse:
                errors[task_id] = np.sqrt(self.model.get_layer(task_id + '-loss')
                                          .eval(session=self.sess, feed_dict=feed_dict))
            elif loss_type is LossTypes.cross_entropy:
                predictions = tf.argmax(self.model.get_layer(task_id + '-prediction'), 1)
                targets = tf.argmax(self.model.get_layer(task_id + '-ground-truth'), 1)
                correct_predictions = tf.equal(predictions, targets)
                accuracy_tensor = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                accuracy = accuracy_tensor.eval(session=self.sess, feed_dict=feed_dict)
                errors[task_id] = 1. - accuracy
        return errors


if __name__ == '__main__':
    model_file = sys.argv[1]
    model_class = LowLevelSharingModel
    task_ids = {Labels.hotness.value: LossTypes.mse,
                Labels.tempo.value: LossTypes.mse,
                Labels.loudness.value: LossTypes.mse}


    evaluation = EvaluateModel(task_ids)
    evaluation.load_data()
    evaluation.load_model(model_file, model_class)
    errors = evaluation.evaluate_model()
    sys.stderr.write("------\n")
    sys.stderr.write("Testing Errors: " + str(errors) + "\n")
