import numpy as np

class Activation:
    def calc(x):
        pass
    def calc_derivative(x):
        pass

class Relu(Activation):
    def calc(x):
        return np.maximum(x, 0)

    def calc_derivative(x):
        return np.where(x <= 0, 0, 1)

class Linear(Activation):
    def calc(x):
        return x

    def calc_derivative(x):
        return 1