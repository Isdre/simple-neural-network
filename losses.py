import numpy as np

class Loss:
    def loss(y_pred,y_true):
        pass
    def loss_derivative(y_pred,y_true):
        pass

class SquareError(Loss):
    def loss(y_pred,y_true):
        return np.square(y_pred - y_true)
    def loss_derivative(y_pred,y_true):
        return 2*(y_pred - y_true)