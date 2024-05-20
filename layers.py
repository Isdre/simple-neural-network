import math
from functools import reduce
import numpy as np

from activation import *

class Layer:
    def __init__(self,neurons_count, activation:Activation, input_shape=None):
        if neurons_count is None: raise Exception("Require to receive amount of neurons")
        else: self.neurons_count = neurons_count

        if input_shape is None:
            self.input_shape = None
        else:
            self.input_shape = list(input_shape)

        self.activation = activation

        self.weights = None
        self.bias = np.random.normal()

    #Creates weight's matrix
    def create_weights_matrix(self,input_shape=None):
        if self.input_shape is None and input_shape is None: raise Exception("Unknown input's shape")
        elif input_shape is not None: self.input_shape = list(input_shape)

        self.weights = np.random.normal(size=(self.neurons_count, reduce(lambda x,y: x*y,self.input_shape)))