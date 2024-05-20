import numpy as np

class Loss:
    name = ""
    def loss(y_pred,y_true):
        pass
    def loss_derivative(y_pred,y_true):
        pass

class SquareError(Loss):
    name = "square_error"
    def loss(y_pred,y_true):
        return np.square(y_pred - y_true)
    def loss_derivative(y_pred,y_true):
        return 2*(y_pred - y_true)