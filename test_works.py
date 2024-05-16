from sklearn import datasets

from layer import Layer
from network import Network


def test_digit_works():
    dataset = datasets.load_digits()
    network = Network()
    network.add(Layer(neurons_count=16,activation="relu",input_shape=[64]))
    network.add(Layer(neurons_count=16, activation="relu"))
    network.add(Layer(neurons_count=10, activation="relu"))
    network.compile(cost="squared_error",metric="accuracy")
    print(dataset.data.shape)
    network.fit(dataset.data,dataset.target)


def test_iris_works():
    pass