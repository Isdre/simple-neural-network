import numpy as np

class Activation:
    name = ""
    def calc(x):
        pass
    def calc_derivative(x):
        pass

    def sigma2(fanIn,fanOut):
        pass

class Relu(Activation):
    name = "relu"
    def calc(x):
        return np.maximum(x, 0)

    def calc_derivative(x):
        return np.where(x <= 0, 0, 1)

    def sigma2(fanIn,fanOut):
        return 2/fanIn

class Linear(Activation):
    name = "linear"
    def calc(x):
        return x

    def calc_derivative(x):
        return 1

    def sigma2(fanIn,fanOut):
        return 2 / (fanIn+fanOut)

class Sigmoid(Activation):
    name = "sigmoid"
    def calc(x):
        return 1 / (1 + np.exp(-x))

    def calc_derivative(x):
        return Sigmoid.calc(x) * (1 - Sigmoid.calc(x))

    def sigma2(fanIn,fanOut):
        return 2 / (fanIn+fanOut)