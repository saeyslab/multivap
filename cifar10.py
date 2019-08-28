import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Input
import numpy as np

num_classes = 10

def get_optimizer():
    return 'adam'

def load_datasets():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train.astype(np.float64), y_train, x_test.astype(np.float64), y_test

def create_model():
    input_shape = (32, 32, 3)
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(inp)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(.2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(.2)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(.2)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    return Model(inputs=inp, outputs=x)
