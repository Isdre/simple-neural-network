import numpy as np

class Loss:
    name = ""
    def loss(self,y_pred,y_true):
        raise NotImplementedError()
    def loss_derivative(self,y_pred,y_true):
        raise NotImplementedError()

class SquareError(Loss):
    name = "square_error"
    def loss(self,y_pred,y_true):
        return np.square(y_pred - y_true)
    def loss_derivative(self,y_pred,y_true):
        return 2*(y_pred - y_true)