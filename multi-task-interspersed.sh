#!/bin/bash
#SBATCH -J MultiInterspersedLearning   # job name
#SBATCH -o multi-interspersed_log.o%j  # output and error file name (%j expands to jobID)
#SBATCH -n 1                           # total number of gpu nodes requested
#SBATCH -p gpu                         # queue (partition) -- normal, development, etc.
#SBATCH -t 10:00:00                    # run time (hh:mm:ss) - 12 hours
#SBATCH --mail-user=tyler@cs.utexas.edu
#SBATCH --mail-type=begin              # email me when the job starts
#SBATCH --mail-type=end                # email me when the job finishes

python multi-task-interspersed.py > Experiments/multi-task-interspersed.txt