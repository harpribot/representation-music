import argparse

from utils.network_utils.params import LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
from utils.training_utils.params import CHECKPOINT_FREQ, EVALUATION_FREQ

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''Train a Multi-Task deep neural network. Note: Only high-level hyperparameters can be set 
        through command-line arguments. Network architechture; and choice of activation function, regularization 
        technique, and iterative optimization procedure can be tweaked by modifying the code.''')

    parser.add_argument('--learning-rate', nargs='?', type=float, default=LEARNING_RATE, const=LEARNING_RATE,
                        help="Learning rate for SGD", dest='learning_rate')
    parser.add_argument('--batch-size', nargs='?', type=float, default=BATCH_SIZE, const=BATCH_SIZE,
                        help="Mini-batch size for SGD-based training", dest='batch_size')
    parser.add_argument('--num-epochs', nargs='?', type=int, default=NUM_EPOCHS, const=NUM_EPOCHS,
                        help="Number of epochs -- number of passes over the entire training set.", dest='num_epochs')
    parser.add_argument('--evaluation-freq', nargs='?', type=int, default=EVALUATION_FREQ, const=EVALUATION_FREQ,
                        help="""Number of mini-batch-SGD steps after which the trained network should be evaluated on 
                        the training and validation steps.""", dest='evaluation_freq')
    parser.add_argument('--checkpoint-freq', nargs='?', type=int, default=CHECKPOINT_FREQ, const=EVALUATION_FREQ,
                        help="""Number of mini-batch-SGD steps after which the trained network should be 
                        checkpointed.""", dest='checkpoint_freq')

    results = parser.parse_args()
    return results