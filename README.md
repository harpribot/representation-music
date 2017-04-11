# representation-music
![msongs](/images/MillionSongs.jpg)

This project explores Multi-Task Representation Learning using shared-architectures for Deep Neural Networks. Experiments are performed to evaluate the utility of training deep neural networks on multiple tasks simulatenously. The aim is to explore four the following four research questions:

- Is multi-task learning beneficial?
- How does training data size impact the efficacy of multi-task learning?
- How does task coupling -- tightly-coupled versus loosely-coupled tasks -- impact the performance of multi-task learning?
- How does the degree of sharing impact performance, and how is this linked to task coupling? (not explored in the current set of experiments)

## Contents
 - [Dataset](#dataset)
 - [Deep Neural Network](#deep-neural-network)
 - [Experimental Results](#experimental-results)
 - [Run Instructions](#run-instructions)
 - [Documentation](#documentation)
 - [Report](#report)
 
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
Otherwise, look into that script and the connecting script to run the experiments. The architecture of experiments and the models is pretty straight forward and highly documented.

A dummy experiment on a synthetic dataset with three tasks using a low-level sharing network can be run by invoking ```maverick_scripts/dummy.sh```.

To evaluate a saved model on the test set, run a command similar to the last command in ```maverick_scripts/dummy.sh``` corresponding to the model-type being evaluated.

## Documentation
The documentation was created automatically, and thus can be error prone. Please report any by raising an issue. Some methods have missing documentation. This is not an error, but laziness on our part. We will add those documentations, when we get some free time.

To access documentation, just open index.html located at
```
docs/build/html/index.html
```
on your favorite browser. You can open them locally for now. I will try hosting them on Github pages once i get time. 

## Report
A write-up for this project is available as `report.pdf`.
