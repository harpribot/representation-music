#!/bin/bash

cd maverick_scripts

echo 'Running the loose buddy...'
sbatch multi-task-low-loose.sh
sbatch multi-task-high-loose.sh
sbatch multi-task-interspersed-loose.sh

echo 'Running the tight friend...'
sbatch multi-task-low-tight.sh
sbatch multi-task-high-tight.sh
sbatch multi-task-interspersed-tight.sh

echo 'Running the loner...'
sbatch single-task-target.sh
sbatch single-task-dependent1.sh
sbatch single-task-dependent2.sh
sbatch single-task-dependent3.sh
sbatch single-task-dependent4.sh
