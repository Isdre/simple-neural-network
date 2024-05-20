import tensorflow as tf
import numpy as np

from layers import Layer
from network import Network

from sklearn.metrics import accuracy_score

def test_digit_tensorflow():
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    #
    # assert X_train.shape == (60000, 28, 28)
    # assert X_test.shape == (10000, 28, 28)
    #
    # network = Network()
    # network.add(Layer(neurons_count=128,activation="relu",input_shape=[28,28]))
    # network.add(Layer(neurons_count=10, activation="relu"))
    # network.compile(loss="mean_squared_error",metric="accuracy")
    #
    # network.fit(X_train,y_train,epochs=3, validation_split=0.1)
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )
    #
    # model.fit(X_train,y_train,epochs=3)
    #
    # y_pred = network.hard_predict(X_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # print(y_pred[:3])
    # y_pred.tofile('y_network_test.csv', sep = ',')
    # print(accuracy_score(y_test, y_pred))
    # y_pred = np.argmax(model.predict(X_test),axis=1)
    # print(y_pred[:3])
    # y_pred.tofile('y_model_test.csv', sep = ',')
    # print(accuracy_score(y_test,y_pred))
    # print(y_test[:3])
    # y_test.tofile('y_test.csv', sep = ',')
    pass

def test_iris_tensorflow():
    pass

if __name__ == "main":
    test_digit_tensorflow()