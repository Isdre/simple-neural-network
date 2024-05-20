import itertools

import numpy as np
from sklearn.model_selection import train_test_split

from layers import Layer
from losses import *
from metric import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.metric = None

        self.functions = None
        self.derivatives_weights = None
        self.derivatives_neurons = None
        self.derivatives_bias = None

        self.__y = None
        self.__gradient_vector = None

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights(input_shape=[self.layers[-2].weights.shape[0]])
        else:
            self.layers[-1].create_weights()

    def compile(self,loss:Loss,metric:Metric):
        self.loss = loss
        self.metric = metric

        self.functions = [None] * (len(self.layers)+1)
        self.derivatives_bias = [None] * (len(self.layers))
        self.derivatives_weights = [None] * sum(l.weights.shape[0] * l.weights.shape[1] for l in self.layers)
        self.derivatives_neurons = [None] * (len(self.layers))

        self.functions[0] = lambda x: x.flatten()
        for i in range(1, len(self.functions)):
            self.functions[i] = (lambda l: lambda x: self.layers[l - 1].activation.calc(np.dot(self.layers[l-1].weights, self.functions[l-1](x)) +
                                                                                        self.layers[l-1].bias))(i)


        #biasses derivatives
        self.derivatives_bias[-1] = lambda x: sum(self.layers[-1].activation.calc_derivative(np.dot(self.layers[-1].weights, self.functions[-2](x))+self.layers[-1].bias) *
                                               self.loss.loss_derivative(self.functions[-1](x), self.__y).mean())


        for i in range(len(self.derivatives_bias) - 2, -1, -1):
            self.derivatives_bias[i] = (lambda l: lambda x: sum(self.derivatives_bias[l+1](x) *
                                                            self.layers[l].activation.calc_derivative(np.dot(self.layers[l].weights, self.functions[l](x)) +
                                                                                                      self.layers[l].bias)))(i)

        #weights derivatives
        self.derivatives_neurons[-1] = lambda x,i: self.layers[-1].activation.calc_derivative(x) * self.loss.loss_derivative(self.functions[-1](x), self.__y)[i]
        for i in range(len(self.layers)-2,-1,-1):
            self.derivatives_neurons[i] = (lambda l: lambda x,n: self.layers[l].weights[:,n] * np.array([self.derivatives_neurons[l+1](x,j) for j in range(n)]) *
                                                                     self.layers[l].activation.calc_derivative(x))(i)

        total_weights_i = 0
        for i in range(len(self.layers)):
            for j in range(self.layers[i].weights.shape[0]):
                for k in range(self.layers[i].weights.shape[1]):
                    self.derivatives_weights[total_weights_i] = lambda x: sum(self.layers[i].weights[j,k] * self.derivatives_neurons[i](x,j))
                    total_weights_i += 1

        #gradient vector
        self.__gradient_vector = lambda x: np.array([i(x) for i in itertools.chain(self.derivatives_weights, self.derivatives_bias)])



    def fit(self,X,y,epochs=1,validation_split=0.1,validation_data=None,validation_target=None,learning_rate=0.01):
        if epochs < 1: raise Exception(f"Epochs can't be less than 1")
        #Prepare y's values
        num_classes = max(y) + 1

        target = np.zeros((len(y), num_classes))

        for idx, label in enumerate(y):
            target[idx, label] = 1

        y = target

        print("Creating validation data")
        if validation_data is None and validation_split == 0.0:
            X_val = X
            y_val = y
            X_train = X
            y_train = y
        elif validation_data is None:
            X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=validation_split,random_state=1905)
        elif validation_target is not None:
            X_train = X
            X_val = validation_data
            y_train = y
            y_val = validation_target
        else:
            raise Exception("validation_data is not None and validation_target is None")

        best_model_w = [np.copy(l.weights)  for l in self.layers]
        best_model_b = [np.copy(l.bias) for l in self.layers]

        y_pred = self.predict(X_val)
        mask = y_pred == np.max(y_pred, axis=1, keepdims=True)
        result = np.zeros_like(y_pred)
        result[mask] = 1

        best_metric = self.metric.calc(y_pred,y_val)

        for epoch in range(epochs):
            print(f"Starting epoch number {epoch}")
            for i in range(len(self.layers)):
                self.layers[i].create_weights()

            for data,target in zip(X_train,y_train):
                self.__y = target
                step = self.__gradient_vector(data) * learning_rate
                total_weights_i = 0

                for i in range(len(self.layers)):
                    for j in range(self.layers[i].weights.shape[0]):
                        for k in range(self.layers[i].weights.shape[1]):
                            self.layers[i].weights[j,k] = step[total_weights_i]
                            total_weights_i += 1

                for i in range(len(self.layers)):
                    self.layers[i].bias = step(total_weights_i + i)

            y_pred = self.predict(X_val)
            actual_metric = self.metric.calc(y_pred,y_val)

            print(f"Ended epoch number {epoch} with {self.metric.name} = {actual_metric}")

            if actual_metric > best_metric:
                best_metric = actual_metric
                best_model_w = [np.copy(l.weights) for l in self.layers]
                best_model_b = [np.copy(l.bias) for l in self.layers]


        for i in range(len(self.layers)):
            self.layers[i].weights = best_model_w[i]
            self.layers[i].bias = best_model_b[i]

    def predict(self,X):
        if np.prod(X.shape) == self.layers[0].weights.shape[1]:
            return self.functions[-1](X)
        elif np.prod(X.shape[1:]) == self.layers[0].weights.shape[1]:
            return np.array([self.functions[-1](x) for x in X])
        else:
            raise Exception(f"X doesn't have correct shape. Layer 0th shape = {self.layers[0].weights.shape}")