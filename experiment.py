import DenseNet
import numpy as np
import keras.backend as K

from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
import getData

def RunDenseNet(batch_size, nb_epoch, depth, nb_dense_block, nb_filter, growth_rate, dropout_rate, weight_decay):
    (X_train, Y_train), (X_test, Y_test), nb_classes, img_dim = getData.getData();
    model = DenseNet.DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate, weight_decay)
    model.summary();
    """Paper Suggests using SGD"""
    opt = SGD(lr=0.0, momentum = 0.9, nesterov = True, decay= weight_decay)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    """Custom learning Schedule"""
    lrs = LearningRateScheduler(custom_LR, verbose=1)
    print("Training")
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks = [lrs], verbose=2)
    print("Evaluating")
    scores = model.evaluate(X_test, Y_test, batch_size=64, verbose=2)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])
    model.save("model.h5")


"""Learning Schedule. Divides lr by 10 after exceeding %50 and %75 of Epochs"""
def custom_LR(epoch):
    lr = 0.1 #UPDATE LEARNING RATE HERE
    if (epoch >= int(0.5 * 25)): #MANUALLY UPDATE EPOCH COUNT HERE
        lr = lr/10
    if (epoch >= int(0.75 * 25)): #MANUALLY UPDATE EPOCH COUNT HERE
        lr = lr/10
    return lr


