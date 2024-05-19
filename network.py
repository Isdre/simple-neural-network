import numpy as np
from sklearn.model_selection import train_test_split

from layers import Layer


class Network:
    def __mean_squared_error(y_pred,y_true):
        return np.mean(np.square(np.round(y_pred - y_true,decimals=3)))

    def __mean_squared_error_derivative(y_pred,y_true):
        return np.mean(2*(np.round(y_pred - y_true,decimals=3)))

    __losses = {
        "mean_squared_error": [__mean_squared_error,__mean_squared_error_derivative]
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

    def __start(x):
        return x

    def __init__(self):
        self.layers = []
        self.loss = None
        self.metric = None
        self.derivatives_weights = None
        self.derivatives_bias = None
        self.derivatives_previous = None
        self.functions = None

    def add(self,layer: Layer):
        self.layers.append(layer)
        if len(self.layers) != 1:
            self.layers[-1].create_weights_matrix(input_shape=[self.layers[-2].weights.shape[0]])
        else:
            self.layers[-1].create_weights_matrix()

    def compile(self,loss,metric):
        if loss not in Network.__losses.keys(): raise Exception(f"Doesn't recognize \"{loss}\" as a cost")
        else:
            self.loss = Network.__losses[loss][0]
            self.loss_derivative = Network.__losses[loss][1]
        if metric not in Network.__metrics.keys(): raise Exception(f"Doesn't recognize \"{metric}\" as a metric")
        else: self.metric = Network.__metrics[metric]

        self.__compare_metric = Network.__compare_metrics[metric]

        self.functions = [] * (len(self.layers) + 1)
        self.functions[0] = Network.__start

        for i,layer in enumerate(self.layers):
            self.functions[i+1] = lambda x: layer.activation(layer.weights * self.functions[i](x) + layer.bias)

        self.derivatives_bias = [] * (len(self.layers) + 1)
        self.derivatives_bias[0] = Network.__start

        for i, layer in enumerate(self.layers):
            self.derivatives_bias[i + 1] = lambda x: layer.derivative(layer.weights * self.functions[i](x) + layer.bias)

        self.derivatives_previous = [] * (len(self.layers) + 1)
        self.derivatives_previous[0] = Network.__start

        for i, layer in enumerate(self.layers):
            self.derivatives_previous[i + 1] = lambda x: layer.weights * layer.derivative(layer.weights * self.functions[i](x) + layer.bias)

        self.derivatives_weights = [] * (len(self.layers) + 1)
        self.derivatives_weights[0] = Network.__start

        for i,layer in enumerate(self.layers):
            self.derivatives_weights[i+1] = lambda x: self.functions[i](x) * layer.derivative(layer.weights * self.functions[i](x) + layer.bias)



    def fit(self,X,y,epochs=1,validation_split=0.0,validation_data=None,validation_target=None,learning_rate=0.01):
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

