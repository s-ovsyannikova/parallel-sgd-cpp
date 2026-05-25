# Parallel Stochastic Gradient Descent in C++

C++ implementation of mini-batch stochastic gradient descent with OpenMP parallelization.  
The project includes a sequential baseline, a parallel version, loss tracking, speedup measurements, and a short mathematical report.

## Overview

The goal of this project is to study the parallelization of stochastic gradient descent for a synthetic regression problem. The implementation compares sequential mini-batch SGD with an OpenMP-based parallel version, where gradient computation inside each mini-batch is distributed across multiple threads.

The dataset is generated synthetically for a regression task:

```text
y = Xw + 5 * sin(x_0) + noise
```

The optimized objective is mean squared error.

### Features
- Sequential mini-batch SGD baseline
- Parallel mini-batch SGD with OpenMP
- Synthetic data generation for regression
- MSE loss computation
- Loss history export to CSV
- Experiments with different batch sizes and thread counts
- Speedup extraction from experiment logs
- 3D speedup surface visualization

### Repository structure
```text
.
├── sgd.cpp
├── run_exps.sh
├── collect_speedup.sh
├── plot_speedup.py
├── losses.zip
├── results.zip
└── report.pdf
```

### Build
The implementation requires a C++ compiler with OpenMP support.
```bash
g++ -O2 -fopenmp sgd.cpp -o sgd_omp_test
```

### Run a single experiment
```bash
export OMP_NUM_THREADS=8
./sgd_omp_test 1024
```
The command runs both sequential and parallel SGD with batch size 1024.

### Run all experiments
```bash
bash run_exps.sh
```
The script evaluates the implementation using:

- batch sizes: ```256```, ```512```, ```1024```, ```2048```, ```4096```, ```8192```
- thread counts: ```1```, ```2```, ```4```, ```8```, ```16```, ```32```, ```56```
Experiment logs are saved to ```results/```.
