import numpy as np
from sklearn.model_selection import train_test_split

from layers import Layer


class Network:
    def __square_error(y_pred,y_true):
        return np.square(np.round(y_pred - y_true,decimals=3))

    def __absolute_error(y_pred, y_true):
        return np.abs(y_pred - y_true)

    __costs = {
        "squared_error": __square_error,
        "absolute_error": __absolute_error,
    }

    def __accuracy(y_pred, y_true):
        return np.all(np.equal(y_pred, y_true),axis=1).mean()

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
            self.layers[-1].create_weights_matrix(input_shape=[self.layers[-2].weights.shape[0]])
        else:
            self.layers[-1].create_weights_matrix()

    def compile(self,cost,metric):
        if cost not in Network.__costs.keys(): raise Exception(f"Doesn't recognize \"{cost}\" as a cost")
        else: self.cost = Network.__costs[cost]
        if metric not in Network.__metrics.keys(): raise Exception(f"Doesn't recognize \"{metric}\" as a metric")
        else: self.metric = Network.__metrics[metric]

        self.__compare_metric = Network.__compare_metrics[metric]

    def fit(self,X,y,epochs=1,validation_split=0.0,validation_data=None,validation_target=None):
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

        print("Created validation data")
        best = self.get_weights()
        best_metric = self.metric(self.soft_predict(X_val),y_val)

        for i in range(epochs):
            print(f"Starting {i+1}th epoch")
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

            actual_metric = self.metric(self.hard_predict(X_val),y_val)
            print(f"Ended {i+1}th epoch: {self.metric.__name__}: {actual_metric}")
            if self.__compare_metric(best_metric,actual_metric):
                best = self.get_weights()

        for w,layer in zip(best,self.layers):
            layer.weights = w

    def soft_predict(self,X):
        y_pred = []

        if self.layers[0].input_shape == list(X.shape):
            a_0 = X
            a_1 = None

            for layer in self.layers:
                a_1 = layer.calc(a_0)
                a_0 = a_1

            y_pred.append(a_1)
        else:
            for x in X:
                a_0 = x
                a_1 = None

                for layer in self.layers:
                    a_1 = layer.calc(a_0)
                    a_0 = a_1

                y_pred.append(a_1)

        return np.array(y_pred)

    def hard_predict(self,X):
        y_pred = self.soft_predict(X)
        if self.layers[0].input_shape == list(X.shape):
            a = np.zeros(y_pred.shape)
            a[:, np.argmax(y_pred)] = 1
        else:
            a = np.zeros(y_pred.shape)
            a[:, np.argmax(y_pred, axis=1)] = 1
        return a

    def get_weights(self):
        w = []
        for layer in self.layers:
            w.append(layer.weights)

        return w
