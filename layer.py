import math

import numpy
import numpy as np
from functools import reduce
import random


class Layer:
    def __relu(x):
        return np.maximum(x, 0)

    def __linear(x):
        return x

    def __gelu(x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    _activation_functions = {"linear" : __linear,
                             "relu" : __relu,
                             "gelu" : __gelu}

    def __init__(self,**kwargs):
        if kwargs["neurons_count"] is None: raise Exception("Require to receive amount of neurons")
        else: self.neurons_count = kwargs["neurons_count"]

        self.input_shape = set(kwargs["input_shape"])

        if kwargs["activation"] is None: raise Exception("Require to receive activation function")
        elif kwargs["activation"] not in Layer._activation_functions.keys(): raise Exception(f"Doesn't recognize \"{kwargs['activation']}\" as a activation function")
        else: self.activation = Layer._activation_functions[kwargs["activation"]]

        self.weights = None

    #Creates weight's matrix
    def create_weights_matrix(self,**kwargs):
        if self.input_shape is None and kwargs["input_shape"] is None: raise Exception("Unknown input's shape")
        elif self.input_shape is None: self.input_shape = set(kwargs["input_shape"])
        self.weights = np.random.normal(size=(self.neurons_count, reduce(lambda x,y: x*y,self.input_shape)))

    #Returns activation_function(weights * X)
    def calc(self,X:numpy.array):
        if set(X.shape) != self.input_shape: raise Exception("Incorrect X's shape")

        if len(X.shape) != 1: X = X.reshape((self.weights.shape[1]))

        return self.activation(np.matmul(self.weights, X))

