import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

def getData():
    """Pre-Process Data"""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    nb_classes = 10;
    img_dim = (32, 32, 3)
    n_channels = X_train.shape[-1]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    """Normalize Data"""
    X = np.vstack((X_train, X_test))
    for i in range(n_channels):
        mean = np.mean(X[:, :, :, i])
        std = np.std(X[:, :, :, i])
        X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
        X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    return (X_train, Y_train), (X_test, Y_test), nb_classes, img_dim

