# WRADMts

Repository for our paper, "Warping resilient Robust Anomaly Detection for Multivariate Time Series" .
This combines two modules 
1) Warp Aligning Temporal Transformation (diffeomorphic transformations on time series to align similar signals)
2) Graph Learning module (to learn temporal and intermetric dependencies)



## Installation
We recommend installing a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via Anaconda.
For instance:
```
conda create -n wradmts
```
Or you can create docker container by following instructions given [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and work on it

## Get Started

1. Install required packages by running
pip install -r requirements.txt

2. Download data. SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/). Other datasets are given in data folder on each modules separately. All datasets are preprocessed and this can be done by executing.
```
python /scripts/process_wadi.py
```
Data format for graph learn module
```

data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)

3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:

## Run
```
    # using gpu
    bash ./scripts/<dataset>.sh <gpu_id> <dataset>

    # or using cpu
    bash ./scripts/<dataset>.sh cpu <dataset>
```
You can change running parameters in  each .sh file.
