import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.utils import to_categorical
import numpy as np
import scipy.io

num_classes = 10

def get_optimizer():
    return 'adam'

def load_datasets():
    # the data, split between train and test sets
    train = scipy.io.loadmat('../datasets/svhn/train.mat')
    x_train = train['X'].transpose((3, 0, 1, 2))
    y_train = train['y']

    test = scipy.io.loadmat('../datasets/svhn/test.mat')
    x_test = test['X'].transpose((3, 0, 1, 2))
    y_test = test['y']

    # filthy hack to fix the white-box batch size
    x_test = x_test[:-2, :, :, :]
    y_test = y_test[:-2, :]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train - 1, num_classes)
    y_test = keras.utils.to_categorical(y_test - 1, num_classes)

    return x_train.astype(np.float64), y_train, x_test.astype(np.float64), y_test

def create_model():
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
