'''
A dummy experiment with synthetic data using low-level sharing.

Tasks:

-- Three dummy tasks

Model:

-- Low Level Sharing Four Hidden Layers

'''
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Experiments.experiment import Experiment
from Models.low_level_sharing_four_hidden import LowLevelSharingModel
from utils.argument_parser import parse_arguments
from utils.data_utils.data_handler import fetch_data
from utils.data_utils.labels import Labels
from utils.network_utils.params import LossTypes
from utils.data_utils.data_handler import convert_to_one_hot

EXPERIMENT_NAME = "dummy-expt"

if __name__ == '__main__':
    args = parse_arguments()

    task_ids = {'1': LossTypes.mse, '2': LossTypes.mse, '3': LossTypes.cross_entropy}
    input_dimension = 5000  # Dimensionality of each training set
    num_inputs_train = 750
    num_inputs_validate = 100
    num_inputs_test = 150

    # Training set
    x_train = np.random.random((num_inputs_train, input_dimension))
    y_train = {}

    # Validation set
    x_validate = np.random.random((num_inputs_validate, input_dimension))
    y_validate = {}

    # Testing set
    x_test = np.random.random((num_inputs_test, input_dimension))
    y_test = {}

    for task_id, loss_type in task_ids.iteritems():
        if loss_type is LossTypes.mse:
            y_train[task_id] = np.random.random((num_inputs_train, 1))
            y_validate[task_id] = np.random.random((num_inputs_validate, 1))
            y_test[task_id] = np.random.random((num_inputs_test, 1))
        elif loss_type is LossTypes.cross_entropy:
            # Training labels -- 2-dimensional one-hot vectors for each example.
            labels = np.random.binomial(1, 0.8, num_inputs_train).reshape(1, num_inputs_train)
            y_train[task_id] = convert_to_one_hot(labels)

            # Validation labels -- 2-dimensional one-hot vectors for each example.
            labels = np.random.binomial(1, 0.8, num_inputs_validate).reshape(1, num_inputs_validate)
            y_validate[task_id] = convert_to_one_hot(labels)

            # Testing labels -- 2-dimensional one-hot vectors for each example.
            labels = np.random.binomial(1, 0.8, num_inputs_test).reshape(1, num_inputs_test)
            y_test[task_id] = convert_to_one_hot(labels)

    exp = Experiment(expt_name=EXPERIMENT_NAME, task_ids=task_ids, x_train=x_train, x_validate=x_validate,
                     x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                     model_class=LowLevelSharingModel, learning_rate=args.learning_rate,
                     batch_size=args.batch_size, num_epochs=args.num_epochs)
    exp.initialize_network()
    exp.train()
