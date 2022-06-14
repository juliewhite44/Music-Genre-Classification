from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, BatchNormalization, Input, add
from keras.models import Model
from keras.regularizers import L2


def res_block(x, filters, stride, regularization_factor=0.0001):
    shortcut = x

    x = BatchNormalization()(x)
    for_shortcut = Activation("relu")(x)
    x = Conv2D(filters//4, (1, 1), use_bias=False, kernel_regularizer=L2(regularization_factor))(for_shortcut)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters//4, (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=L2(regularization_factor))(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (1, 1), use_bias=False, kernel_regularizer=L2(regularization_factor))(x)

    if stride[0] > 1:
        shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=L2(regularization_factor))(for_shortcut)

    x = add([x, shortcut])
    return x


def get_ResNet(shape, classes, stages, filters, regularization_factor=0.0001):
    inputs = Input(shape=shape)
    x = BatchNormalization()(inputs)
    x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=L2(regularization_factor))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for i in range(0, len(stages)):
        x = res_block(x, filters[i + 1], (2, 2))

        for j in range(0, stages[i] - 1):
            x = res_block(x, filters[i + 1], (1, 1))

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D((8, 8))(x)

    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=L2(regularization_factor))(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x, name="MPUM_ResNet")

    return model
