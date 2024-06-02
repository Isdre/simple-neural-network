import itertools
import random

import numpy as np
from sklearn.model_selection import train_test_split

from layers import Layer
from losses import *
from metric import *
from optimizers import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.metric = None
        self.optimizer = None

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights(input_shape=[self.layers[-2].weights.shape[0]])
        else:
            self.layers[-1].create_weights()

    def compile(self, loss:Loss, optimizer:Optimizer, metric:Metric):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric



    def fit(self,X,y,epochs=1,batch_size=1,validation_split=0.3,validation_data=None,validation_target=None):
        if epochs < 1: raise Exception(f"Epochs can't be less than 1")
        #Prepare y's values
        num_classes = max(y) + 1

        target = np.zeros((len(y), num_classes))

        for idx, label in enumerate(y):
            target[idx, label] = 1

        y = target

        print("Preparing data")
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

        best_model_w = [l.weights.copy() for l in self.layers]

        y_pred = self.predict(X_val)
        print(y_pred)
        best_metric = self.metric.calc(y_pred,y_val)


        for epoch in range(epochs):
            print(f"Starting epoch number {epoch}")
            actual_metric = self.__fit_for_epoch(X_train, y_train, batch_size, X_val, y_val)
            print(f"Ended epoch number {epoch} with loss = {self.loss.loss(y_pred,y_val).mean()}, {self.metric.name} = {actual_metric}")

            if self.metric.compare(actual_metric,best_metric):
                best_metric = actual_metric
                best_model_w = [l.weights.copy() for l in self.layers]

        for i in range(len(self.layers)):
            self.layers[i].weights = best_model_w[i].copy()

        print(f"Best {self.metric.name} = {best_metric}")

    def __fit_for_epoch(self,X_train,y_train,batch_size,X_val,y_val):

        for i in range(len(self.layers)):
            self.layers[i].create_weights()

        pairs = list(zip(X_train, y_train))
        random.shuffle(pairs)

        for data, target in pairs:
            # Forwardpropagation
            a = [data.flatten()]
            for l in self.layers:
                a_1 = l.activation.calc(np.dot(l.weights, a[-1]) + l.bias)
                a.append(a_1)

            # Backpropagation
            loss = self.loss.loss(a[-1], target)
            self.optimizer.optimize(self.layers, a, target, loss)

        y_pred = self.predict(X_val)
        # print(y_pred)
        return self.metric.calc(y_pred, y_val)

    def predict(self,X):
        if np.prod(X.shape) == self.layers[0].weights.shape[1]:
            a = X.flatten()
            for l in self.layers:
                a = l.activation.calc(np.dot(l.weights, a))
            a = np.array(a)
            max_values = np.max(a, keepdims=True)
            return np.where(a == max_values, 1, 0)
        elif np.prod(X.shape[1:]) == self.layers[0].weights.shape[1]:
            y_pred = []
            for x in X:
                a = x.flatten()
                for l in self.layers:
                    a = l.activation.calc(np.dot(l.weights, a))
                y_pred.append(a)
            y_pred = np.array(y_pred)
            max_values = np.max(y_pred, axis=1, keepdims=True)
            return np.where(y_pred == max_values, 1, 0)
        else:
            raise Exception(f"X doesn't have correct shape. Layer 0th shape = {self.layers[0].weights.shape}")