# Introduction

This repository contains a collection of Python scripts and Jupyter notebooks to implement spatial multivariate imputation using Machine learning and Lambda distribution.

# Python Setup

It is recommended to use Anaconda Python distribution and create an environment using the provided `environment.yml` file.

# Notebooks

The notebooks use the Python scripts that are located in `Tools` folder.

- `original.dat` is the input data file.
- `01-DataInventory.ipynb` the first notebook that takes care of data processing and exporting the data for imputation.
- `02-LambdaDistributionMlModel.ipynb` this notebook is used to fit the lambda distribution and its results will be used by downstream notebooks.
- `03-MlForConditionalDistributionTemplate.ipynb` a template notebook to implement training of the neural networks. It is not required to run this notebook as it is used by the next notebook to execute training.
- `04-DataImputationUsingMLP.ipynb` the main notebook that implements all the steps of imputation and cross validation. This notebook uses `papermill` python package to orchestrate training of neural networks.
