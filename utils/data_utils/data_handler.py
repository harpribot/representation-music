from Data.msd import MillionSongDataset
from params import FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS
from utils.training_utils.params import TRAIN_FRACTION, TEST_FRACTION, VALIDATE_FRACTION, TOTAL_NUM_EXAMPLES
import numpy as np
from utils.training_utils.params import EXPT_DIRECTORY_PATH
from utils.network_utils.params import LossTypes
import os
import sys


def fetch_data(task_ids):
    """
    Fetches the dataset from the Database and then divides it into training / testing / validation data
    :param task_ids: Dictionary of task identifiers-loss type pairs indexed by task-id.
    :return: split data, and labels
    """
    # Load the data set in memory.
    db = MillionSongDataset(FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS)
    # Generate train/validation/test splits.
    db.generate_split(TRAIN_FRACTION, VALIDATE_FRACTION, TEST_FRACTION, TOTAL_NUM_EXAMPLES)

    sys.stderr.write("------\n")
    # Training set
    sys.stderr.write("Creating training set\n")
    x_train = np.array([bow for bow in db.get_bow(db.train)], dtype=float)
    labels_train = np.array([t.vector() for t in db.get_features(db.train)])

    # Validation set
    sys.stderr.write("Creating validation set\n")
    x_validate = np.array([bow for bow in db.get_bow(db.validate)], dtype=float)
    labels_val = np.array([t.vector() for t in db.get_features(db.validate)])

    # Testing set
    sys.stderr.write("Creating testing set\n")
    x_test = np.array([bow for bow in db.get_bow(db.test)], dtype=float)
    labels_test = np.array([t.vector() for t in db.get_features(db.test)])

    # Close databases to free memory.
    db.close()

    # Build the label dictionary using tasks under consideration, discarding other labels.
    y_train = {}
    y_val = {}
    y_test = {}
    for task_id, loss_type in task_ids.iteritems():
        if loss_type is LossTypes.mse:
            y_train[task_id] = np.array(labels_train[:, int(task_id)], dtype=float).reshape(-1, 1)
            y_val[task_id] = np.array(labels_val[:, int(task_id)], dtype=float).reshape(-1, 1)
            y_test[task_id] = np.array(labels_test[:, int(task_id)], dtype=float).reshape(-1, 1)
        elif loss_type is LossTypes.cross_entropy:
            # Training labels -- 2-dimensional one-hot vectors for each example.
            labels = np.array(labels_train[:, int(task_id)], dtype=int).reshape(1, -1)
            y_train[task_id] = convert_to_one_hot(labels)

            # Validation labels -- 2-dimensional one-hot vectors for each example.
            labels = np.array(labels_val[:, int(task_id)], dtype=int).reshape(1, -1)
            y_val[task_id] = convert_to_one_hot(labels)

            # Testing labels -- 2-dimensional one-hot vectors for each example.
            labels = np.array(labels_test[:, int(task_id)], dtype=int).reshape(1, -1)
            y_test[task_id] = convert_to_one_hot(labels)

    return x_train, x_validate, x_test, y_train, y_val, y_test


def create_experiment(expt_name):
    """
    Creates a directory for this experiment where all the logs and outputs are saved. The directory is named
    after the experiment-name suffixed with a 4-digit random number.

    :param expt_name: Experiment name.
    :return: None
    """
    os.chdir(EXPT_DIRECTORY_PATH)  # Change working directory to the Experiments folder
    expt_dir = expt_name + "-" + str(np.random.randint(1000, 9999))
    try:
        os.makedirs(expt_dir)
    except OSError:
        if not os.path.isdir(expt_dir):
            raise
    finally:
        os.chdir("./" + expt_dir)  # Change working directory to the newly created folder
        sys.stderr.write("------\n")
        sys.stderr.write("Experiment Directory: " + expt_dir + "\n")


def convert_to_one_hot(labels):
    num_examples = labels.shape[1]
    one_hot_vectors = np.zeros((num_examples, 2))
    one_hot_vectors[np.arange(num_examples), labels] = 1
    return one_hot_vectors
