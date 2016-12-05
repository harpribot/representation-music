from enum import Enum

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 1000


class LossTypes(Enum):
    mse = "mse"
    cross_entropy = "cross-entropy"
