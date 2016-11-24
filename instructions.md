How to run train the model
==========================
cd to the project directory and execute `python experiment.py`


How to test the pipeline before using the MSD?
==============================================
Run the "dummy" experiment that uses a small synthetic dataset to run the pipeline. Call the `dummy` method in the `__name__ == '__main__'` condition at the bottom of experiment.py. Use small values for `NUM_EPOCHS` (5), `FREQ_OF_EVALUATIONS` (5), and `FREQ_OF_CHECKPOINTS` (10) in params.py to run this experiment. You should see periodic evaluation results on the screen, and model dumps and plots in a new directory created under the Experiments folder.

Note: The plotting function is not yet tested -- I can't get matplotlib to work on my machine! It might throw errors which can, hopefully, be fixed easily.

Changes to be done before running an experiment
================================================
- Global variables in the utils/network_utils/params.py and utils/training_utils/params.py file
- `task_ids` in main method in experiment.py
- `expt_name` argument passed to the constructor of `Experiment` class: Read the documentation for its description
- `model_class` argument to the constructor of the `Experiment` class: The model to be used for the experiment

```
However I think it would be better if we use argparse to create all the arguments that we can feed to the experiments
folder, instead of creating a params.py folder and changing it all the time.
```
