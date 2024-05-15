import numpy
import numpy as np
from functools import reduce
import random


class Layer:
    _activation_functions_names = ["linear", "relu", "gelu"]

    def __init__(self,**kwargs):
        if kwargs["neurons_count"] is None: raise Exception("Require to receive amount of neurons")
        else: self.neurons_count = kwargs["neurons_count"]

        self.input_shape = set(kwargs["input_shape"])

        if kwargs["activation"] is None: raise Exception("Require to receive activation function")
        elif kwargs["activation"] not in Layer._activation_functions_names: raise Exception(f"Doesn't recognize \"{kwargs['activation']}\" as a activation function")
        else: self.activation = kwargs["activation"]

        self.weights = None

    #Creates weight's matrix
    def create_weights_matrix(self,**kwargs):
        if self.input_shape is None and kwargs["input_shape"] is None: raise Exception("Unknown input's shape")
        elif self.input_shape is None: self.input_shape = set(kwargs["input_shape"])
        self.weights = np.random.normal(size=(self.neurons_count, reduce(lambda x,y: x*y,self.input_shape)))

    #Returns weights * X
    def calc(self,X:numpy.array):
        if set(X.shape) != self.input_shape: raise Exception("Incorrect X's shape")

        if len(X.shape) != 1: X = X.reshape((self.weights.shape[1]))

        return np.matmul(self.weights, X)


