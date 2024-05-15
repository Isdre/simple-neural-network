import numpy as np

from layer import Layer


class Network:

    def __init__(self):
        self.layers = []
        pass

    def add(self,layer: Layer):
        self.layers.append(layer)