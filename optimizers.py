import numpy as np

class Optimizer:
    def __init__(self,learning_rate,epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def optimize(self,layers,a,target,loss):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, learning_rate=1e-06,epsilon=1e-07,momentum=0.0):
        super().__init__(learning_rate,epsilon)
        self.__momentum = momentum

        if self.__momentum < 0 or self.__momentum > 1:
            raise ValueError("Momentum must be between 0 and 1")

        self.__weights_prev = None
        self.__bias_prev = None
    def optimize(self,layers,a,target,loss):
        if self.__weights_prev is None:
            self.__weights_prev = [np.zeros(layer.weights.shape) for layer in layers]
        if self.__bias_prev is None:
            self.__bias_prev = [np.zeros(layer.bias.shape) for layer in layers]
        y_pred = a[-1]
        delta = y_pred - target

        # delta[delta < 0] = -1
        # delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)

            self.__weights_prev[i] = self.__momentum * self.__weights_prev[i] + (1 - self.__momentum) * step
            self.__bias_prev[i] = self.__momentum * self.__bias_prev[i] + (1 - self.__momentum) * delta

            self.__weights_prev[i][(self.__weights_prev[i] < self.epsilon) & (self.__weights_prev[i] > -1 * self.epsilon)] = 0
            self.__bias_prev[i][(self.__bias_prev[i] < self.epsilon) & (self.__bias_prev[i] > -1 * self.epsilon)] = 0



            layers[i].weights -= self.learning_rate * self.__weights_prev[i]
            layers[i].bias -= self.learning_rate * self.__bias_prev[i]

            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])

class RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-06, epsilon=1e-07, beta=0.999):
        super().__init__(learning_rate,epsilon)
        self.__beta = beta

        if self.__beta < 0 or self.__beta > 1:
            raise ValueError("Beta must be between 0 and 1")

        self.__s_weights_prev = None
        self.__s_bias_prev = None

    def optimize(self,layers,a,target,loss):
        if self.__s_weights_prev is None:
            self.__s_weights_prev = [np.zeros(layer.weights.shape) for layer in layers]
        if self.__s_bias_prev is None:
            self.__s_bias_prev = [np.zeros(layer.bias.shape) for layer in layers]

        y_pred = a[-1]
        delta = y_pred - target

        # delta[delta < 0] = -1
        # delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)
            self.__s_weights_prev[i] =(self.__beta * self.__s_weights_prev[i] + (1 - self.__beta) * (step ** 2)).mean()
            self.__s_bias_prev[i] = (self.__beta * self.__s_bias_prev[i] + (1 - self.__beta) * (delta ** 2)).mean()

            layers[i].weights -= self.learning_rate * (step / (np.sqrt(self.__s_weights_prev[i])+self.epsilon))
            layers[i].bias -= self.learning_rate * (delta / (np.sqrt(self.__s_bias_prev[i])+self.epsilon))

            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])

class Adam(Optimizer):
    def __init__(self, learning_rate=1e-06, epsilon=1e-07,momentum=0.9, beta=0.999):
        super().__init__(learning_rate,epsilon)
        self.__momentum = momentum
        self.__beta = beta

        if self.__momentum < 0 or self.__momentum > 1:
            raise ValueError("Momentum must be between 0 and 1")
        if self.__beta < 0 or self.__beta > 1:
            raise ValueError("Beta must be between 0 and 1")

        self.__weights_prev = None
        self.__bias_prev = None
        self.__s_weights_prev = None
        self.__s_bias_prev = None

    def optimize(self,layers,a,target,loss):
        if self.__weights_prev is None:
            self.__weights_prev = [np.zeros(layer.weights.shape) for layer in layers]
        if self.__bias_prev is None:
            self.__bias_prev = [np.zeros(layer.bias.shape) for layer in layers]
        if self.__s_weights_prev is None:
            self.__s_weights_prev = [np.zeros(layer.weights.shape) for layer in layers]
        if self.__s_bias_prev is None:
            self.__s_bias_prev = [np.zeros(layer.bias.shape) for layer in layers]

        y_pred = a[-1]
        delta = y_pred - target

        # delta[delta < 0] = -1
        # delta[delta > 0] = 1

        delta = delta * loss

        for i in range(len(layers) - 1, -1, -1):
            a_i = a[i]
            step = np.outer(delta, a_i)

            self.__weights_prev[i] = self.__momentum * self.__weights_prev[i] + (1 - self.__momentum) * step
            self.__bias_prev[i] = self.__momentum * self.__bias_prev[i] + (1 - self.__momentum) * delta

            self.__weights_prev[i][(self.__weights_prev[i] < self.epsilon) & (self.__weights_prev[i] > -1 * self.epsilon)] = 0
            self.__bias_prev[i][(self.__bias_prev[i] < self.epsilon) & (self.__bias_prev[i] > -1 * self.epsilon)] = 0

            self.__s_weights_prev[i] = (self.__beta * self.__s_weights_prev[i] + (1 - self.__beta) * (step ** 2)).mean()
            self.__s_bias_prev[i] = (self.__beta * self.__s_bias_prev[i] + (1 - self.__beta) * (delta ** 2)).mean()

            weights_corrected = self.__weights_prev[i] / (1 - self.__momentum)
            bias_corrected = self.__bias_prev[i] / (1 - self.__momentum)
            s_weights_corrected = self.__s_weights_prev[i] / (1 - self.__beta)
            s_bias_corrected = self.__s_bias_prev[i] / (1 - self.__beta)

            layers[i].weights -= self.learning_rate * (weights_corrected / (np.sqrt(s_weights_corrected)+self.epsilon))
            layers[i].bias -= self.learning_rate * (bias_corrected / (np.sqrt(s_bias_corrected)+self.epsilon))

            if i != 0:
                delta = np.dot(layers[i].weights.T, delta) * layers[i - 1].activation.calc_derivative(a[i])