import math
from functools import reduce

import numpy
import numpy as np


class Layer:
    def __relu(x):
        return np.maximum(x, 0)

    def __relu_derivative(x):
        return np.where(x<=0,0,1)

    def __linear(x):
        return x

    def __linear_derivative(x):
        return 1

    __activation_functions = {"linear" : __linear,
                             "relu" : __relu}

    __activation_derivative = {"linear": __linear,
                              "relu": __relu}

    def __init__(self,neurons_count, activation, input_shape=None):
        if neurons_count is None: raise Exception("Require to receive amount of neurons")
        else: self.neurons_count = neurons_count

        if input_shape is None:
            self.input_shape = None
        else:
            self.input_shape = list(input_shape)

        if activation is None: raise Exception("Require to receive activation function")
        elif activation not in Layer.__activation_functions.keys(): raise Exception(f"Doesn't recognize \"{activation}\" as a activation function")
        else: self.activation = Layer.__activation_functions[activation]

        self.derivative = Layer.__activation_derivative[activation]
        self.weights = None
        self.bias = 0

    #Creates weight's matrix
    def create_weights_matrix(self,input_shape=None):
        if self.input_shape is None and input_shape is None: raise Exception("Unknown input's shape")
        elif input_shape is not None: self.input_shape = list(input_shape)

        self.weights = np.random.normal(size=(self.neurons_count, reduce(lambda x,y: x*y,self.input_shape)))