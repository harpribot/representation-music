from Data.msd import MillionSongDataset
from params import FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS
from utils.training_utils.params import TRAIN_FRACTION, TEST_FRACTION, VALIDATE_FRACTION, TOTAL_NUM_EXAMPLES
import numpy as np
from utils.training_utils.params import EXPT_DIRECTORY_PATH
import os


def fetch_data(task_ids):
    """
    Fetches the dataset from the Database and then divides it into training / testing / validation data
    :param task_ids: labels on which the network is trained
    :return: split data, and labels
    """
    # Load the data set in memory.
    db = MillionSongDataset(FEATURES, LYRICS, LYRIC_MAPPINGS, TRACKS)
    # Generate train/validation/test splits.
    db.generate_split(TRAIN_FRACTION, VALIDATE_FRACTION, TEST_FRACTION, TOTAL_NUM_EXAMPLES)

    print("------")
    # Training set
    print("Creating training set")
    x_train = np.array([bow for bow in db.get_bow(db.train)], dtype=float)
    labels_train = np.array([t.vector() for t in db.get_features(db.train)])

    # Validation set
    print("Creating validation set")
    x_validate = np.array([bow for bow in db.get_bow(db.validate)], dtype=float)
    labels_val = np.array([t.vector() for t in db.get_features(db.validate)])

    # Testing set
    print("Creating testing set")
    x_test = np.array([bow for bow in db.get_bow(db.test)], dtype=float)
    labels_test = np.array([t.vector() for t in db.get_features(db.test)])

    # Build the label dictionary using tasks under consideration, discarding other labels.
    y_train = {task_id: np.array(labels_train[:, int(task_id)], dtype=float).reshape(-1, 1) for task_id in task_ids}
    y_val = {task_id: np.array(labels_val[:, int(task_id)], dtype=float).reshape(-1, 1) for task_id in task_ids}
    y_test = {task_id: np.array(labels_test[:, int(task_id)], dtype=float).reshape(-1, 1) for task_id in task_ids}

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
        print("------")
        print("Experiment Directory: " + expt_dir)
