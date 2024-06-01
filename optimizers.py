import numpy as np

class Optimizer:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def optimize(self,layers,a,delta):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, learning_rate=0.00001):
        super().__init__(learning_rate)
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
    def __init__(self, learning_rate=0.00000001, beta1=0.9, beta2=0.999, epsilon=0.00000001):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon=epsilon
        self.momentum = None
        self.v = None
        self.t = 0


    def optimize(self,layers,a,target,loss):
        if self.momentum is None:
            self.momentum = [np.zeros(layer.weights.shape) for layer in layers]
        if self.v is None:
            self.v = [np.zeros(layer.weights.shape) for layer in layers]

        self.t += 1
        y_pred = a[-1]
        delta = y_pred - target

        delta[delta < 0] = -1
        delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)
            self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * step
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (step ** 2)

            m_hat = self.momentum[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            layers[i].weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])
