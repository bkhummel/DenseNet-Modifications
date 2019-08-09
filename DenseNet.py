from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K


"""Function called in Main, Builds DenseNet"""
def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None, weight_decay=1E-4):
    concat_axis = -1
    nb_layers = 5
    """Build Input Layer, Including first Convolution"""
    inputLayer = Input(img_dim)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False)(inputLayer)

    """Build DenseBlocks and Transition Layers"""
    for i in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = transition(x, concat_axis=concat_axis, nb_filter=nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    """ Final DenseBLock and Output Layer """
    x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes, activation='softmax')(x)

    """Return Constructed Keras Model"""
    densenet = Model(inputs=[inputLayer], outputs=[x], name="DenseNet")
    return densenet



"""Builds DenseBlock with 5 Densely connected Convolutional Blocks"""
def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    inputBlock = [x]
    for i in range(nb_layers):
        x = convBlock(x, concat_axis,growth_rate,dropout_rate, weight_decay)
        inputBlock.append(x)
        x = Concatenate(axis=concat_axis)(inputBlock)
        nb_filter += growth_rate
    return x, nb_filter


"Builds Convolutional Blocks, these are layers in denseblock"
def convBlock(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


"""Builds Transition Layers"""
def transition(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):

    """BatchNormalization"""
    x = BatchNormalization(axis=concat_axis)(x)

    """Convolution"""
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),kernel_initializer="he_uniform", padding="same",use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    """Pooling"""
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

