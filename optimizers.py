import numpy as np

class Optimizer:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def optimize(self,layers,a,delta):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, learning_rate=0.000002):
        super(learning_rate)
    def optimize(self,layers,a,target,loss):
        y_pred = a[-1]
        delta = y_pred - target

        delta[delta < 0] = -1
        delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)
            layers[i].weights -= self.learning_rate * step
            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])

class Adam(Optimizer):
    def __init__(self, learning_rate=0.000002, alpha=0.8, beta=0.8):
        super(learning_rate)
        self.alpha = alpha
        self.beta = beta

    def optimize(self,layers,a,target,loss):
        y_pred = a[-1]
        delta = y_pred - target

        delta[delta < 0] = -1
        delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)
            layers[i].weights -= self.learning_rate * step
            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])
