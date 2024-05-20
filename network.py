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

        self.y = None
        self.__gradient_vector = None

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights_matrix(input_shape=[self.layers[-2].weights.shape[0]])
        else:
            self.layers[-1].create_weights_matrix()

    def compile(self,loss:Loss,metric:Metric):
        self.loss = loss
        self.metric = metric

        self.functions = [None] * (len(self.layers)+1)
        self.derivatives_bias = [None] * (len(self.layers))
        self.derivatives_weights = [None] * sum(l.weights.shape[0] * l.weights.shape[1] for l in self.layers)

        self.functions[0] = lambda x: x
        for i in range(1, len(self.functions)):
            self.functions[i] = (lambda l: lambda x: self.layers[l - 1].activation(np.dot(self.layers[l - 1].weights, self.functions[l - 1](x)) + self.layers[l - 1].bias))(i)

        self.derivatives_bias[-1] = lambda x: self.layers[-1].derivative(np.dot(self.layers[-1].weights, self.functions[-2](x))) * np.mean(self.loss_derivative(self.functions[-1](x), self.y))

        for i in range(len(self.derivatives_bias) - 2, -1, -1):
            self.derivatives_bias[i] = (lambda l: lambda x: self.derivatives_bias[l + 1](x) * self.layers[l].derivative(np.dot(self.layers[l].weights, self.functions[l - 1](x))))(i)

        total_i = 0




    def fit(self,X,y,epochs=1,validation_split=0.0,validation_data=None,validation_target=None,batch=1,learning_rate=0.01):
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

    def predict(self,X):
        if X.shape == self.layers[0].shape:
            return self.functions[-1](X)
        elif X.shape[1:] == self.layers[0].shape:
            return np.apply_along_axis(self.functions[-1],axis=1,arr=X)
        else:
            raise Exception(f"X doesn't have correct shape {X.shape} != {self.layers[0].shape} and {X.shape[1:]} != {self.layers[0].shape}")