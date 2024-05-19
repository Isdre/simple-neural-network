import tensorflow as tf
import keras
import numpy as np

from layer import Layer
from network import Network


def test_digit_tensorflow():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)

    print(X_train.shape)

    network = Network()
    network.add(Layer(neurons_count=16,activation="relu",input_shape=[28,28]))
    network.add(Layer(neurons_count=16, activation="relu"))
    network.add(Layer(neurons_count=10, activation="relu"))
    network.compile(cost="squared_error",metric="accuracy")

    network.fit(X_train,y_train,epochs=3, validation_split=0.1)



    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, validation_split=0.1, epochs=3)

    print(network.metric(network.hard_predict(X_test),y_test))
    print(model.evaluate(X_test, y_test))

    print(y_test[0])

def test_iris_tensorflow():
    pass

if __name__ == "main":
    test_digit_tensorflow()