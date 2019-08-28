import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from PIL import Image
import numpy as np
import zipfile

WIDTH, HEIGHT = 64, 64
SIZE = 12499
input_shape = (WIDTH, HEIGHT, 3)

def get_optimizer():
    return 'adam'

def load_datasets():
    images = []
    labels = []
    with zipfile.ZipFile('/home/jpeck/datasets/asirra.zip', 'r') as f:
        for i in range(SIZE):
            with f.open('PetImages/Cat/{}.jpg'.format(i)) as g:
                try:
                    image = Image.open(g)
                    images.append(np.array(image.resize((WIDTH, HEIGHT)).getdata()).reshape(input_shape).astype(np.float64))
                    labels.append([1., 0.])
                except:
                    pass
            with f.open('PetImages/Dog/{}.jpg'.format(i)) as g:
                try:
                    image = Image.open(g)
                    images.append(np.array(image.resize((WIDTH, HEIGHT)).getdata()).reshape(input_shape).astype(np.float64))
                    labels.append([0., 1.])
                except:
                    pass
    images, labels = np.array(images), np.array(labels)

    idx = int(.8*images.shape[0])
    x_train, y_train = images[:idx], labels[:idx]
    x_test, y_test = images[idx:], labels[idx:]

    return x_train, y_train, x_test, y_test

def create_model():
    input_shape = (WIDTH, HEIGHT, 3)
    num_classes = 2
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
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
