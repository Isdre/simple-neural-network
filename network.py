import numpy as np
from sklearn.model_selection import train_test_split

from layer import Layer


class Network:
    def __square_error(y_pred,y_true):
        return np.square(y_pred - y_true)

    def __absolute_error(y_pred, y_true):
        return np.abs(y_pred - y_true)

    __costs = {
        "squared_error": __square_error,
        "absolute_error": __absolute_error,
    }

    def __accuracy(y_pred, y_true):
        return np.equal(y_pred, y_true).mean()

    __metrics = {
        "accuracy": __accuracy,
    }

    #return True if b is better than a
    def __compare_accuracy(a,b):
        return b > a

    __compare_metrics = {
        "accuracy": __compare_accuracy,
    }

    def __init__(self):
        self.layers = []
        self.cost = None
        self.metric = None

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights_matrix(input_shape=self.layers[-1].weights.shape[0])
        else:
            self.layers[-1].create_weights_matrix()

    def compile(self,cost,metric):
        if cost not in Network.__costs.keys(): raise Exception(f"Doesn't recognize \"{cost}\" as a cost")
        else: self.cost = Network.__costs[cost]
        if metric not in Network.__metrics.keys(): raise Exception(f"Doesn't recognize \"{metric}\" as a metric")
        else: self.metric = Network.__metrics[metric]

        self.__compare_metric = Network.__compare_metrics[self.metric]

    def fit(self,X,y,epochs=1,validation_split=0.0,validation_data=None,validation_target=None):
        if epochs < 1: raise Exception(f"Epochs can't be less than 1")

        if validation_data is None and validation_split == 0.0:
            X_val = X
            y_val = y
            X_train = X
            y_train = y
        elif validation_data is None:
            X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=validation_split)
        elif validation_target is not None:
            X_train = X
            X_val = validation_data
            y_train = y
            y_val = validation_target
        else:
            raise Exception("validation_data is not None and validation_target is None")


        best = self.get_weights()
        best_metric = self.metric(self.predict(X_val),y_val)

        for i in range(epochs):
            a = []
            for data,target in zip(X_train,y_train):
                a_0 = data
                a_1 = None

                for layer in self.layers:
                    a.append(a_0)
                    a_1 = layer.calc(a_0)
                    a_0 = a_1

                c = self.cost(a_1,target)

                for layer in reversed(self.layers):
                    c = layer.learn(a[-1],c)
                    a.pop(-1)

            actual_metric = self.metric(self.predict(X_val),y_val)
            if self.__compare_metric(best_metric,actual_metric):
                best = self.get_weights()

    def predict(self,X):
        y_pred = np.array([])
        for x in X:
            a_0 = x
            a_1 = None

            for layer in self.layers:
                a_1 = layer.calc(a_0)
                a_0 = a_1

            y_pred = np.vstack([y_pred, a_1])

        return y_pred

    def get_weights(self):
        w = []
        for layer in self.layers:
            w.append(layer.weights)

        return w
