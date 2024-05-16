import numpy as np

from layer import Layer


class Network:

    def __init__(self):
        self.layers = []

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights_matrix(input_shape=self.layers[-1].weights.shape[0])
        else:
            self.layers[-1].create_weights_matrix()

