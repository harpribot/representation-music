# Data set split
TRAIN_FRACTION = 0.45
VALIDATE_FRACTION = 0.10
TEST_FRACTION = 0.45
TOTAL_NUM_EXAMPLES = 60000
# TOTAL_NUM_EXAMPLES = 147052     # Total number of tracks
assert(TRAIN_FRACTION + VALIDATE_FRACTION + TEST_FRACTION == 1.0)

# File paths
EXPT_DIRECTORY_PATH = "./Experiments"  # Path of the Experiments directory
EXPT_NAME = "default"
