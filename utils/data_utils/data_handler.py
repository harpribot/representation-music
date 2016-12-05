import os
import sys
import numpy as np
from sklearn import preprocessing

from Data.msd import MillionSongDataset
from params import FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS
from utils.data_utils.labels import Labels
from utils.network_utils.params import LossTypes
from utils.training_utils.params import TRAIN_FRACTION, TEST_FRACTION, VALIDATE_FRACTION, TOTAL_NUM_EXAMPLES
from utils.training_utils.params import EXPT_DIRECTORY_PATH


def fetch_data(tasks):
    """
    Fetches the dataset from the Database and then divides it into training / testing / validation data
    :param task_ids: Dictionary of task identifiers-loss type pairs indexed by task-id.
    :return: split data, and labels
    """
    # Load the data set in memory.
    db = MillionSongDataset(FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS)
    # Generate train/validation/test splits.
    db.generate_split(TRAIN_FRACTION, VALIDATE_FRACTION, TEST_FRACTION, TOTAL_NUM_EXAMPLES)

    # Actual names of the tasks (e.g., 'pop' or 'loudness')
    task_labels = tasks.keys()
    # Assign integer identifiers each label that serve as an indices into feature vector.
    task_mappings = Labels(task_labels)
    task_ids      = [task_mappings.get(t) for t in task_labels]
    # Recreate input dictionary using integer labels for each task.
    tasks = { str(task_mappings.get(t)) : tasks[t] for t in tasks.keys() }
    # Print task mappings.
    for i in range(len(task_ids)):
        sys.stderr.write('%s : %s\n' % (task_labels[i], task_ids[i]))

    sys.stderr.write("------\n")
    # Training set
    sys.stderr.write("Creating training set\n")
    x_train = np.array([bow for bow in db.get_bow(db.train)], dtype=float)
    labels_train = np.array([t.vector(task_labels) for t in db.get_features(db.train)])

    # Validation set
    sys.stderr.write("Creating validation set\n")
    x_validate = np.array([bow for bow in db.get_bow(db.validate)], dtype=float)
    labels_val = np.array([t.vector(task_labels) for t in db.get_features(db.validate)])

    # Testing set
    sys.stderr.write("Creating testing set\n")
    x_test = np.array([bow for bow in db.get_bow(db.test)], dtype=float)
    labels_test = np.array([t.vector(task_labels) for t in db.get_features(db.test)])

    x_train, x_validate, x_test = standardize_input(x_train, x_validate, x_test)
    labels_train, labels_val, labels_test = standardize_labels(labels_train, labels_val, labels_test, task_ids)

    # Close databases to free memory.
    db.close()

    # Build the label dictionary using tasks under consideration, discarding other labels.
    y_train = {}
    y_val = {}
    y_test = {}
    for task_id, loss_type in tasks.iteritems():
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

    return x_train, x_validate, x_test, y_train, y_val, y_test, tasks


def standardize_input(train, validate, test):
    """
    Standardizes the input by converting all features to zero-mean, unit-variance
    :param train: Training input set
    :param validate: Validation input set
    :param test: Test input set
    :return: Standardized train set, Standardized validation set, Standardized test set.
    """
    num_train = train.shape[0]
    num_validate = validate.shape[0]
    num_test = test.shape[0]

    stacked = np.vstack((train, validate, test))
    standardized = preprocessing.scale(stacked, axis=0)

    start = 0
    end = num_train
    standardized_train = standardized[start: end, :]

    start += end
    end += num_validate
    standardized_validate = standardized[start: end, :]

    start += end
    standardized_test = standardized[start:, :]
    return standardized_train, standardized_validate, standardized_test


def standardize_labels(train, validate, test, task_ids):
    """
    Standardizes the labels by converting all real numbered labels to zero-mean, unit-variance
    :param train: Training label set
    :param validate: Validation label set
    :param test: Test label set
    :param task_ids: Dictionary of task identifiers-loss type pairs indexed by task-id.
    :return: Standardized train set, Standardized validation set, Standardized test set.
    """
    num_train = train.shape[0]
    num_validate = validate.shape[0]
    num_test = test.shape[0]

    stacked = np.vstack((train, validate, test))

    # A temporary column of zeros. Will be removed later.
    standardized = np.zeros(num_train + num_validate + num_test).reshape(-1, 1)
    for task_id, loss_type in task_ids.iteritems():
        task_labels = stacked[:, int(task_id)]
        if loss_type is LossTypes.mse:
            task_standardized = preprocessing.scale(task_labels, axis=0)
        else:
            task_standardized = task_labels
        standardized = np.hstack((standardized, task_standardized))

    # Remove the stray all-zero first column.
    standardized = standardized[1:, :]
    start = 0
    end = num_train
    standardized_train = standardized[start: end, :]

    start += end
    end += num_validate
    standardized_validate = standardized[start: end, :]

    start += end
    standardized_test = standardized[start:, :]
    return standardized_train, standardized_validate, standardized_test


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
