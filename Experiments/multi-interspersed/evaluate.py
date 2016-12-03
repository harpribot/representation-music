'''
Evaluate the multi-task model with interspersed sharing.

Tasks:

-- Hotness
-- Duration
-- Year

Model:

-- Interspersed Sharing Four Hidden Layers

'''
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Experiments.evaluate_model import EvaluateModel
from Models.interspersed_sharing_four_hidden import InterspersedSharingModel
from utils.data_utils.labels import Labels

# Path of the checkpoint file.
# Should look like "./Experiments/expt-name/epoch-n"
MODEL_FILE = ""

if __name__ == '__main__':
    model_class = InterspersedSharingModel
    task_ids = [Labels.hotness.value, Labels.duration.value, Labels.year.value]

    evaluation = EvaluateModel(task_ids)
    evaluation.load_data()
    evaluation.load_model(MODEL_FILE, model_class)
    errors = evaluation.evaluate_model()
    sys.stderr.write(str(errors) + "\n")
