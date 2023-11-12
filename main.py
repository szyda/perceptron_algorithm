import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs

# sygnaly wejsciowe pomnozyc przez wagi
# sygnal s = sumujemy te wartosci
# bipolarna funkcja aktywacji f
# x<= 0 -> -1
# x > 0 -> 1

def unipolar_function(s):
    return 1 if s > 0 else 0

def generate_points(number_of_points):
    X, y = make_blobs(number_of_points, centers=2)
    return X, y

def generate_weights(number_of_weights):
    return np.zeros(number_of_weights)

def train_perceptron(X, T, weights, num_epochs, number_of_plot):
    number_of_epochs, number_of_types = X.shape

    for epoch in range(num_epochs):
        for i in range(number_of_epochs):
            x_i = X[i] # input
            t_i = T[i] # output

            s_i = np.dot(weights, x_i)
            y_i = unipolar_function(s_i)
            e_i = t_i - y_i

            delta_w = e_i * x_i
            weights = weights + delta_w

        if np.all(T == [unipolar_function(np.dot(weights, x)) for x in X]):
            print(f'Wykres nr {number_of_plot}. Perceptron jest nauczony.')
        else:
            print(f'Nie mozna odseparowac punktow dla wykresu {number_of_plot}')
        break

    return weights


def generate_plot(X, T, w, number_of_plot):
    plt.scatter(X[:, 0], X[:, 1], c=T)

    x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2 = (-w[0] / w[1]) * x1
    plt.plot(x1, x2, '-r', label='Wykres')

    plt.title(f'Wykres nr {number_of_plot}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def main():
    number_of_plots = 5
    for i in range(number_of_plots):
        number_of_points = 20
        X, T = generate_points(number_of_points)
        num_weights = X.shape[1]
        starting_weights = generate_weights(num_weights)
        number_of_epochs = 200

        trained_weights = train_perceptron(X, T, starting_weights, number_of_epochs, i+1)
        print("Wagi po nauczeniu:", trained_weights)
        print("\n")

        generate_plot(X, T, trained_weights, i+1)

main()



