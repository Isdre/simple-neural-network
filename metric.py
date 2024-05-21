import numpy as np

class Metric:
    name = ""
    def compare(a,b):
        raise NotImplementedError()

    def calc(y_pred,y_true):
        raise NotImplementedError()

class Accuracy(Metric):
    name = "accuracy"
    def compare(a,b):
        if a > b: return True

    def calc(y_pred,y_true):
        return np.all(np.equal(y_pred, y_true), axis=1).mean()