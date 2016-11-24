How to train the model
======================
cd to the project directory and execute `python experiment.py`. You can pass command-line arguments to set various hyperparameters; run `python experiment.py -h` for the usage guide. These arguments are optional; if they are passed, the default values specified in utils/network_utils/params.py and utils/training_utils/params.py will be used. Note that low-level hyperparameters such as network architechture; and choice of activation function, regularization technique, and iterative optimization procedure can only be tweaked by modifying the code.


How to test the pipeline before using the MSD?
==============================================
Run the "dummy" experiment that uses a small synthetic dataset to run the pipeline. Call the `dummy` method in the `__name__ == '__main__'` condition at the bottom of experiment.py. Use small values for `--num-epochs` (e.g., 5), `--evaluation-freq` (e.g., 5), and `--checkpoint-freq` (e.g., 10) to run this experiment. You should see periodic evaluation results on the screen, and model dumps and plots in a new directory created under the Experiments folder.

Note: The plotting function is not yet tested -- I can't get matplotlib to work on my machine! It might throw errors which can, hopefully, be fixed easily.

Changes to be done before running an experiment
================================================
- Global variables in the utils/network_utils/params.py and utils/training_utils/params.py file. Some of these can be set through command-line arguments.
- `task_ids` in main method in experiment.py
- `expt_name` argument passed to the constructor of `Experiment` class: Read the documentation for its description
- `model_class` argument to the constructor of the `Experiment` class: The model to be used for the experiment
