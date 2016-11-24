from enum import Enum


# SGD related
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 1000

# Dataset split
TRAIN_FRACTION = 0.75
VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.15
TOTAL_NUM_EXAMPLES = 147052     # Total number of tracks
assert(TRAIN_FRACTION + VALIDATE_FRACTION + TEST_FRACTION == 1)

# Validation and checkpointing
FREQ_OF_EVALUATIONS = 1000   # Number of minibatch GD steps after which an evaluation is performed.
FREQ_OF_CHECKPOINTS = 3000   # Number of minibatch GD steps after which a checkpoint is taken.

# File paths
EXPT_DIRECTORY_PATH = "./Experiments"  # Path of the Experiments directory
FEATURES = './Data/msongs.db'
LYRICS = './Data/mxm_dataset.db'
LYRIC_MAPPINGS = './Data/bow.txt'
TRACKS = './Data/tracks.txt'


# Task related
class Task(Enum):
    hotttnesss = '0'
    duration = '1'
    key = '2'
    loudness = '3'
    year = '4'
    time_signature = '5'
    tempo = '6'
    tags = '7'
