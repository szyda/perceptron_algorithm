# Single perceptron implementation for binary classification
## Overview
This project demonstrates the implementation of a single perceptron to solve the problem of binary classification of two datasets. Utilizing Python, the sklearn library for data generation, and matplotlib for visualization, this repository contains code that models a perceptron's learning process through iterative training and adjustment of weights based on input signals and a unipolar activation function.

## Features
- **Activation function:** Implements a unipolar activation function that outputs 1 for positive input signals and 0 otherwise.
- **Data generation:** Uses `sklearn.datasets.make_blobs` to generate datasets with two centers for binary classification.
- **Weight initialization:** Begins with zero-initialized weights for simplicity and demonstration purposes.
- **Training process:** Adjusts perceptron weights based on the difference between expected output and the perceptron's prediction.
- **Visualization:** Generates plots to visually represent the classification boundary developed by the perceptron over the dataset.

## Implementation
The process involves initializing weights, generating a dataset, and iteratively training the perceptron over a specified number of epochs. The training adjusts weights in response to classification errors, aiming to minimize these errors over time. Once training is complete, the script visualizes the results, showing the data points and the decision boundary defined by the trained weights.

## Dependencies:
**NumPy**
```
pip install numpy
```
**Matplotlib**
```
pip install matplotlib
```
**sklearn datasets**
```
pip install scikit-datasets
```

## How to Run:
1. Clone the repository.
2. Install dependencies.
3. Run the script (python main.py).
<br>
**Note:** This project was made for an Introduction to Artificial Intelligence course at school, showcasing practical application of AI principles in a controlled educational environment.
