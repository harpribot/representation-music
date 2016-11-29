'''
A multi-task baseline experiment using low-level sharing.

Tasks:

-- Hotness
-- Duration
-- Year

Model:

-- Low Level Sharing Four Hidden Layers

'''
from experiment import Experiment
from utils.argument_parser import parse_arguments
from utils.data_utils.labels import Labels
from utils.data_utils.data_handler import fetch_data, create_experiment
from Models.low_level_sharing_four_hidden import LowLevelSharingFourHiddenModel

EXPERIMENT_NAME = "multi-task-low"

if __name__ == '__main__':
    args = parse_arguments()

    # We will predict a single label for this experiment.
    task_ids = [Labels.hotness.value, Labels.duration.value, Labels.year.value]

    # Produce the training, validation, and testing set.
    x_train, x_validate, x_test, y_train, y_validate, y_test = fetch_data(task_ids)

    e = Experiment(expt_name=EXPERIMENT_NAME,
                   task_ids=task_ids,
                   x_train=x_train, x_validate=x_validate, x_test=x_test,
                   y_train=y_train, y_validate=y_validate, y_test=y_test,
                   model_class=LowLevelSharingFourHiddenModel,
                   learning_rate=args.learning_rate,
                   batch_size=args.batch_size,
                   num_epochs=args.num_epochs,
                   checkpoint_freq=args.checkpoint_freq,
                   evaluation_freq=args.evaluation_freq)

    e.initialize_network()
    e.train()
