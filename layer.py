import math
from functools import reduce

import numpy
import numpy as np


class Layer:
    def __relu(x):
        return np.maximum(x, 0)

    def __linear(x):
        return x

    def __gelu(x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    __activation_functions = {"linear" : __linear,
                             "relu" : __relu,
                             "gelu" : __gelu}

    def __init__(self,neurons_count, activation, input_shape=None):
        if neurons_count is None: raise Exception("Require to receive amount of neurons")
        else: self.neurons_count = neurons_count

        if input_shape is None:
            self.input_shape = None
        else:
            self.input_shape = set(input_shape)

        if activation is None: raise Exception("Require to receive activation function")
        elif activation not in Layer.__activation_functions.keys(): raise Exception(f"Doesn't recognize \"{activation}\" as a activation function")
        else: self.activation = Layer.__activation_functions[activation]

        self.weights = None
        self.bias = 0 #make bias

    #Creates weight's matrix
    def create_weights_matrix(self,input_shape=None):
        if self.input_shape is None and input_shape is None: raise Exception("Unknown input's shape")
        elif input_shape is not None: self.input_shape = set(input_shape)
        self.weights = np.random.normal(size=(self.neurons_count, reduce(lambda x,y: x*y,self.input_shape)))

    #Returns activation_function(weights * X)
    def calc(self,X:numpy.array):
        if set(X.shape) != self.input_shape: raise Exception("Incorrect X's shape")

        if len(X.shape) != 1: X = X.reshape((self.weights.shape[1]))

        print(X.shape)
        print(self.weights.shape)

        return self.activation(np.clip(np.matmul(self.weights, X), -1e10, 1e10))

    #Ask Mr. Pięta about it
    def learn(self,prev_a,loss):
        #              nx1  mx1
        a_i = np.outer(loss,prev_a)
        print(f"{prev_a.shape} {loss.shape}")
        print(self.weights.shape)
        #              nxm            mxn
        self.weights = self.weights + a_i
        return np.mean(a_i,axis=0)
