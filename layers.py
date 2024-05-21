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

    #Creates weight's matrix
    def create_weights(self, input_shape=None):
        if self.input_shape is None and input_shape is None: raise Exception("Unknown input's shape")
        elif input_shape is not None: self.input_shape = list(input_shape)

        self.weights = np.random.normal(scale=self.activation.sigma2(np.prod(self.input_shape),self.neurons_count),size=(self.neurons_count, np.prod(self.input_shape)))