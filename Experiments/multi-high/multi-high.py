"""
High-level sharing multi-task experiments used in final report.

Considers two sets of tasks.

Tightly coupled: { 'pop', 'pop rock', 'ballad' }
Loosely coupled: { 'pop', 'loudness', 'year' }

Model

-- High-Level Sharing Task with Four Hidden Layers
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
from Experiments.experiment import Experiment
from Models.high_level_sharing_four_hidden import HighLevelSharingModel
from utils.argument_parser import parse_arguments
from utils.data_utils.data_handler import fetch_data
from utils.training_utils.task_set import Coupled

EXPERIMENT_NAME = 'multi-high-final'

if __name__ == '__main__':
    args = parse_arguments()

    # Target tasks.
    task = Coupled.tasks[args.task_type]
    
    to_run = {('%s-%s' % (EXPERIMENT_NAME, args.task_type)): task}
    
    # These are the training sizes which we will test.
    training_sizes = [500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 25000]
    
    for name, tasks in to_run.iteritems():      
        # Produce the training, validation, and test sets.
        x_train, x_validate, x_test, y_train, y_valid2352ate, y_test, task_ids = fetch_data(tasks)
        
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
            
            e = Experiment(task_ids=task_ids,
                           x_train=this_x_train, x_validate=x_validate, x_test=x_test,
                           y_train=this_y_train, y_validate=y_validate, y_test=y_test,
                           model_class=HighLevelSharingModel,
                           expt_name=expt_name,
                           learning_rate=args.learning_rate,
                           batch_size=args.batch_size,
                           num_epochs=args.num_epochs)
            e.initialize_network()
            e.train()
