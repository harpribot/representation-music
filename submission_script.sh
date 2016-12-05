#!/bin/bash

echo 'Running the loose buddy...'
sbatch maverick_scripts/multi-task-low-loose.sh
sbatch maverick_scripts/multi-task-high-loose.sh
sbatch maverick_scripts/multi-task-interspersed-loose.sh

echo 'Running the tight friend...'
sbatch maverick_scripts/multi-task-low-tight.sh
sbatch maverick_scripts/multi-task-high-tight.sh
sbatch maverick_scripts/multi-task-interspersed-tight.sh

echo 'Running the loner...'
sbatch maverick_scripts/single-task-target.sh
sbatch maverick_scripts/single-task-dependent1.sh
sbatch maverick_scripts/single-task-dependent2.sh
sbatch maverick_scripts/single-task-dependent3.sh
sbatch maverick_scripts/single-task-dependent4.sh
