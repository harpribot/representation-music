'''
Low-level sharing multi-task experiments used in final report.

Considers two sets of tasks.

Tightly coupled: { 'pop', 'pop rock', 'ballad' }
Loosely coupled: { 'pop', 'loudness', 'year' }

Model

-- Low-Level Sharing Task with Four Hidden Layers
'''

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Experiments.experiment import Experiment
from Models.low_level_sharing_four_hidden import LowLevelSharingModel
from utils.argument_parser import parse_arguments
from utils.data_utils.data_handler import fetch_data
from utils.data_utils.labels import Labels
from utils.network_utils.params import LossTypes

EXPERIMENT_NAME = 'multi-low-final'



if __name__ == '__main__':
    args = parse_arguments()

    # Target tasks.
    tightly_coupled = { 'pop'      : LossTypes.cross_entropy,
                        'pop rock' : LossTypes.cross_entropy,
                        'ballad'   : LossTypes.cross_entropy }
    loosely_coupled = { 'pop'      : LossTypes.cross_entropy,
                        'loudness' : LossTypes.mse,
                        'year'     : LossTypes.mse }
    
    to_run = { ('%s-tightly' % (EXPERIMENT_NAME)) : tightly_coupled,
               ('%s-loosely' % (EXPERIMENT_NAME)) : loosely_coupled }

    # These are the training sizes which we will test.
    training_sizes = [500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 25000]
    
    for name, tasks in to_run.iteritems():      
        # Produce the training, validation, and test sets.
        x_train, x_validate, x_test, y_train, y_validate, y_test, task_ids = fetch_data(tasks)
        
        
        for size in training_sizes:
            # Create train sets.
            this_x_train = x_train[:size, :]
            this_y_train = {t_id: y_train[t_id][:size] for t_id in y_train.keys()}
        
            expt_name = ('%s-training%d' % (name, size))
            
            sys.stderr.write('Experiment Name:')
            sys.stderr.write(str(name) + '\n')

            sys.stderr.write('Experiment Tasks:')
            sys.stderr.write(str(tasks) + '\n')

            sys.stderr.write('Training Size:')
            sys.stderr.write(str(size) + '\n')

            e = Experiment(task_ids = task_ids,
                           x_train=this_x_train, x_validate=x_validate, x_test=x_test,
                           y_train=this_y_train, y_validate=y_validate, y_test=y_test,
                           model_class=LowLevelSharingModel,
                           expt_name = expt_name,
                           learning_rate=args.learning_rate,
                           batch_size=args.batch_size,
                           num_epochs=args.num_epochs)
            e.initialize_network()
            e.train()
