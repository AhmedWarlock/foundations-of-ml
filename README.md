# Foundations of Machine Learning

This repository contains the foundational concepts and implementations of machine learning algorithms. It serves as a starting point for understanding and implementing various machine learning algorithms.

Each algorithm is in its own folder named accordingly. The repository is implemented using Python 3.12.3.

## Table of Contents

- [Logistic Regression](#logistic-regression)
- [NN from Scratch](#nn-from-scratch)

## Logistic Regression

The "Logistic Regression" folder contains the implementation and examples of logistic regression, a popular binary classification algorithm. Logistic regression is widely used in machine learning and is a fundamental algorithm for understanding the basics of classification.

In this folder, you will find:

- `data.py`: This file contains code related to data generation and processing.
- `utils.py`: contains utility functions or helper functions used in the logistic regression implementation.
- `model.py`: The model.py file implements the logistic regression model. It includes a class that define the logistic regression algorithm, including the training and prediction steps. It also includes methods for model evaluation and performance metrics.
- `main.py`: acts as the starting point for the logistic regression program. It contains the code responsible for training and testing the logistic regression model using the given datasets. This file coordinates the entire workflow by invoking functions from data.py, utils.py, and model.py to execute the essential steps for logistic regression.

## NN from Scratch

The "NN from Scratch" folder contains the implementation of a neural network from scratch using only basic Python libraries. This folder aims to provide an in-depth understanding of the internals of a neural network and the process of building it from the ground up.

In this folder, you will find:

- `data.py`: This file contains data generation of a random dataset with 800 samples generated  using random.random.randn to generate random numbers from the normal distribution. Also, it includes splitting the data into train and test sets.
- `utils.py`: Contains The activations function and it's derivative and other utility functions.
- `model.py`: Contains the model for the classification algorithm.
- `main.py`.


## Contact

For any questions or inquiries, please contact aabdalla@aimsammi.org.
