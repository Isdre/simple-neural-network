import tensorflow as tf

import activation
from layers import Layer
from network import Network

from losses import *
from metric import *
from activation import *

def test_digit_works():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    network = Network()

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)



    print(X_train.shape)

    network.add(Layer(neurons_count=16,activation=Relu,input_shape=[28,28]))
    network.add(Layer(neurons_count=16, activation=Relu))
    network.add(Layer(neurons_count=10, activation=Relu))
    network.compile(loss=SquareError,metric=Accuracy)

    print(y_test[0:5])
    print(network.predict(X_test[0:5]))
    network.fit(X_train[:5],y_train[:5],epochs=3)
    print(network.predict(X_test[0:5]))
    pass

def test_iris_works():
    pass

if __name__ == "main":
    test_digit_works()