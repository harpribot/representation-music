# Data set split
TRAIN_FRACTION = 0.75
VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.15
TOTAL_NUM_EXAMPLES = 25000
#TOTAL_NUM_EXAMPLES = 147052     # Total number of tracks
assert(TRAIN_FRACTION + VALIDATE_FRACTION + TEST_FRACTION == 1.0)

# Validation and check pointing
EVALUATION_FREQ = 900   # Number of mini-batch SGD steps after which an evaluation is performed.
CHECKPOINT_FREQ = 900   # Number of mini-batch SGD steps after which a checkpoint is taken.

# File paths
EXPT_DIRECTORY_PATH = "./Experiments"  # Path of the Experiments directory
EXPT_NAME = "default"
