# import os
import sys
import tensorflow as tf
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Models.low_level_sharing_four_hidden import LowLevelSharingModel
from utils.data_utils.labels import Labels
from utils.data_utils.data_handler import fetch_data


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
        self.sess = tf.Session()
        self.model = model_class(self.task_ids, self.input_dimension, self.output_dimensions)
        self.model.create_model()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        sys.stderr.write("Model " + model_file + " loaded.\n")

    def load_data(self):
        """
        Loads the test dataset.
        """
        _, _, self.x_test, _, _, self.y_test = fetch_data(self.task_ids)
        self.input_dimension = self.x_test.shape[1]
        self.train_samples = self.x_test.shape[0]
        self.output_dimensions = {task_id: self.y_test[task_id].shape[1] for task_id in self.task_ids}

    def evaluate_model(self):
        """
        Returns a dictionary of errors indexed by task identifiers where each element denotes the error for that
        task on the test set.

        :return dictionary of test errors
        """
        feed_dict = dict()
        feed_dict[self.model.get_layer('input')] = self.x_test
        for id_ in self.task_ids:
            feed_dict[self.model.get_layer(id_ + '-ground-truth')] = self.y_test[id_]
        errors = {}
        for task_id in self.task_ids:
            errors[task_id] = self.model.get_layer(task_id + '-loss').eval(session=self.sess, feed_dict=feed_dict)
        return errors


if __name__ == '__main__':
    model_file = sys.argv[1]
    model_class = LowLevelSharingModel
    task_ids = [Labels.hotness.value, Labels.duration.value, Labels.year.value]

    evaluation = EvaluateModel(task_ids)
    evaluation.load_data()
    evaluation.load_model(model_file, model_class)
    errors = evaluation.evaluate_model()
    sys.stderr.write(str(errors) + "\n")
