import argparse

from utils.network_utils.params import LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
from utils.training_utils.params import EXPT_NAME

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''Train a Multi-Task deep neural network. Note: Only high-level hyperparameters can be set 
        through command-line arguments. Network architechture; and choice of activation function, regularization 
        technique, and iterative optimization procedure can be tweaked by modifying the code.''')

    parser.add_argument('--learning-rate', nargs='?', type=float, default=LEARNING_RATE, const=LEARNING_RATE,
                        help="Learning rate for SGD", dest='learning_rate')
    parser.add_argument('--batch-size', nargs='?', type=int, default=BATCH_SIZE, const=BATCH_SIZE,
                        help="Mini-batch size for SGD-based training", dest='batch_size')
    parser.add_argument('--num-epochs', nargs='?', type=int, default=NUM_EPOCHS, const=NUM_EPOCHS,
                        help="Number of epochs -- number of passes over the entire training set.", dest='num_epochs')
    parser.add_argument('--experiment-name', nargs='?', type=str, default=EXPT_NAME, const=EXPT_NAME,
                        help="""Name of the experiment to be used to name the experiment's directory""",
                        dest='experiment_name')

    results = parser.parse_args()
    return results
