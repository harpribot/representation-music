# representation-music
![msongs](/images/MillionSongs.jpg)

Multi-Task Representation Learning using Shared Architecture for Deep Neural Networks

## Contents
 - [Dataset](#dataset)
 - [Deep Neural Network](#deep-neural-network)
 - [Experimental Results](#experimental-results)
 - [Run Instructions](#run-instructions)
 
## Dataset
The dataset can be obtained from [here](http://labrosa.ee.columbia.edu/millionsong) and the correponding paper is cited below.
```
Bertin-Mahieux, Thierry, et al. "The million song dataset." ISMIR. Vol. 2. No. 9. 2011.
```
## Deep Neural Network
![models](/images/Models.png)

## Experimental Results
#### Performance on Tightly Coupled Task
![plots_tight](/images/main_plot.png)

#### Performance on Loosely Coupled Task
![plots_loose](/images/loosely_plot.png)

## Run Instructions
If you are using a supercomputing node that accepts SLURM jobs, then run the following after editing the scripts in maverick_scripts (by adding your email address instead of the given placeholder):
```
sbatch submission_script.sh
```
else, look into that script and the connecting script to run the experiments. The architecture of experiments and the models is pretty straight forward and highly documented.
