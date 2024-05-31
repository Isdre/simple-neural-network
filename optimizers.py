import numpy as np

class Optimizer:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def optimize(self,layers,a,delta):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, learning_rate=0.000002):
        self.learning_rate = learning_rate
    def optimize(self,layers,a,delta):
        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)
            layers[i].weights -= self.learning_rate * step
            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])
