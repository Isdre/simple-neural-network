import tensorflow as tf

from layer import Layer
from network import Network


def test_digit_works():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    network = Network()

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)

    print(X_train.shape)

    network.add(Layer(neurons_count=16,activation="relu",input_shape=[28,28]))
    network.add(Layer(neurons_count=16, activation="relu"))
    network.add(Layer(neurons_count=10, activation="relu"))
    network.compile(cost="squared_error",metric="accuracy")

    network.fit(X_train,y_train)
    network.hard_predict(X_test[0])

def test_iris_works():
    pass

if __name__ == "main":
    test_digit_works()