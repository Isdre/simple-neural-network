import numpy as np

class Metric:
    name = ""
    def compare(a,b):
        pass

    def calc(y_pred,y_true):
        pass

class Accuracy(Metric):
    name = "accuracy"
    def compare(a,b):
        if a < b: return True

    def calc(y_pred,y_true):
        return np.all(np.equal(y_pred, y_true), axis=1).mean()