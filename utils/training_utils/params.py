# Data set split
TRAIN_FRACTION = 0.75
VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.15
TOTAL_NUM_EXAMPLES = 147052     # Total number of tracks
assert(TRAIN_FRACTION + VALIDATE_FRACTION + TEST_FRACTION == 1)

# Validation and check pointing
FREQ_OF_EVALUATIONS = 1000   # Number of mini-batch GD steps after which an evaluation is performed.
FREQ_OF_CHECKPOINTS = 3000   # Number of mini-batch GD steps after which a checkpoint is taken.

# File paths
EXPT_DIRECTORY_PATH = "./Experiments"  # Path of the Experiments directory
