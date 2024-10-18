import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from layers import *
from losses import *
from metric import *
from activation import *
from optimizers import *
from network import Network

def test_digit_tensorflow():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)

    network = Network()
    network.add(Dense(neurons_count=128, activation=Relu, input_shape=[28, 28]))
    network.add(Dense(neurons_count=128, activation=Relu))
    network.add(Dense(neurons_count=10))
    network.compile(loss=SquareError(), optimizer=Adam(0.0002), metric=Accuracy())

    network.fit(X_train, y_train, epochs=5,batch_size=32, validation_split=0.1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

    y_pred = np.argmax(network.predict(X_train), axis=1)
    print(y_pred[:3])
    # y_pred.tofile('y_network_test.csv', sep=',')
    print(accuracy_score(y_train, y_pred))
    y_pred = np.argmax(model.predict(X_train), axis=1)
    print(y_pred[:3])
    # y_pred.tofile('y_model_test.csv', sep=',')
    print(accuracy_score(y_train, y_pred))

    print(y_train[:3])

    y_pred = np.argmax(network.predict(X_test), axis=1)
    print(y_pred[:3])
    #y_pred.tofile('y_network_test.csv', sep=',')
    print(accuracy_score(y_test, y_pred))

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(y_pred[:3])
    #y_pred.tofile('y_model_test.csv', sep=',')
    print(accuracy_score(y_test, y_pred))
    print(y_test[:3])
    #y_test.tofile('y_test.csv', sep=',')

if __name__ == "__main__":
    test_digit_tensorflow()
