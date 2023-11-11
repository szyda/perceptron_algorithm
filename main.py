import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs

# sygnaly wejsciowe pomnozyc przez wagi
# sygnal s = sumujemy te wartosci
# bipolarna funkcja aktywacji f
# x<= 0 -> -1
# x > 0 -> 1

def bipolar_function(s):
    return 1 if s > 0 else -1

def generate_points(number_of_points):
    return make_blobs(number_of_points, centers=2)[0]

def generate_weights(number_of_weights):
    return np.zeros(number_of_weights)


def learn():
    pass




