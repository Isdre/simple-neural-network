import tensorflow as tf

import activation
from layers import Layer
from network import Network

from losses import *
from metric import *
from activation import *
from optimizers import *

def test_digit_works():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    network = Network()

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)

    network.add(Layer(neurons_count=16,activation=Relu,input_shape=[28,28]))
    network.add(Layer(neurons_count=16, activation=Relu))
    network.add(Layer(neurons_count=10, activation=Linear))
    network.compile(loss=SquareError,optimizer=Adam(), metric=Accuracy)
    print(y_test[0:6])
    print(network.predict(X_test[:5]))
    print(network.predict(X_test[6]))
    network.fit(X_train,y_train,epochs=3,batch_size=1)
    print(network.predict(X_test[:5]))
    print(network.predict(X_test[6]))

if __name__ == "main":
    test_digit_works()