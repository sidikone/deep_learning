import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import List
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D

from tensorflow.keras.utils import to_categorical


class NeuralNetwork:
    def __init__(self, name=None):
        self.neural_network = Sequential(name=name)

    def add_input(self, nb_lines, nb_cols=None):
        if nb_cols is None:
            input_def = Input(shape=(nb_lines,))
        else:
            input_def = Input(shape=(nb_lines, nb_cols))

        self.neural_network.add(input_def)
        return None

    def add_dense(self, nb, activation=None):
        input_def = Dense(units=nb, activation=activation)
        self.neural_network.add(input_def)
        return None

    def add_conv1D(self, filter_nb, filter_siz, padding_type='same', activation=None):
        input_def = Conv1D(filters=filter_nb,
                           kernel_size=filter_siz,
                           padding=padding_type,
                           activation=activation)
        self.neural_network.add(input_def)
        return None

    def add_pooling1D(self, typ='max', siz=2, padding='valid'):

        if typ is 'max':
            input_def = MaxPooling1D(pool_size=siz,
                                     padding=padding)
        elif typ is 'mean':
            input_def = AveragePooling1D(pool_size=siz,
                                         padding=padding)
        self.neural_network.add(input_def)
        return None

    def add_flatten(self):
        self.neural_network.add(Flatten())
        return None

    def compile_network(self, optim, loss, metrics):
        self.neural_network.compile(optimizer=optim,
                                    loss=loss,
                                    metrics=metrics)
        return None

    def fit_network(self, x_train, y_train, validation, epochs, batch_size, verbose=0):
        self.neural_network.fit(x=x_train,
                                y=y_train,
                                validation_data=validation,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose)
        return None

    def evaluate_network(self, x_test, y_test):
        self.neural_network.evaluate(x_test, y_test)
        return None

    def save(self, input_name):
        self.neural_network.save(filepath=input_name)
        return None

    def summary(self):
        self.neural_network.summary()
        return None


def data_loader(typ: str) -> List:
    x_train, x_ctrl = 0, 0
    if typ is 'set_1':
        x_train = np.load('../data/data_set_1/x_train_1.npy')
        x_ctrl = np.load('../data/data_set_1/x_ctrl_1.npy')

    elif typ is 'set_2':
        x_train = np.load('../data/data_set_1/x_train_2.npy')
        x_ctrl = np.load('../data/data_set_1/x_ctrl_2.npy')

    y_train = np.load('../data/data_set_1/y_train.npy')
    y_train = to_categorical(y_train)

    y_ctrl = np.load('../data/data_set_1/y_ctrl.npy')
    y_ctrl = to_categorical(y_ctrl)

    return [x_train, y_train, x_ctrl, y_ctrl]


def first_model(data, save=False):
    net = NeuralNetwork(name='1D_neural_network')
    net.add_input(nb_lines=100)
    net.add_dense(nb=100, activation='relu')
    net.add_dense(nb=50, activation='relu')
    net.add_dense(nb=10, activation='softmax')
    opt_net = optimizers.Adam(learning_rate=1e-2)
    net.compile_network(optim=opt_net, loss='categorical_crossentropy', metrics=['accuracy'])
    net.summary()
    [x_train, y_train, x_ctrl, y_ctrl] = data
    
    net.fit_network(x_train, y_train, validation=(x_ctrl, y_ctrl), batch_size=128, epochs=500, verbose=2)

    if save is not None:
        net.save(save)
    return None


def second_model(data, save=False):
    net2 = NeuralNetwork(name='2D_neural_network')
    net2.add_input(nb_lines=10, nb_cols=10)
    net2.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
    net2.add_pooling1D(typ='max', siz=2)
    net2.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
    net2.add_pooling1D(typ='max', siz=2)
    net2.add_flatten()
    net2.add_dense(nb=10, activation='softmax')
    net2.summary()

    [x_train, y_train, x_ctrl, y_ctrl] = data
    opt_net = optimizers.Adam(learning_rate=1e-2)
    net2.compile_network(optim=opt_net, loss='categorical_crossentropy', metrics=['accuracy'])
    net2.fit_network(x_train, y_train, validation=(x_ctrl, y_ctrl), batch_size=128, epochs=500, verbose=2)

    if save is not None:
        net2.save(save)
    return None


def main():
    data_1 = data_loader('set_1')
    first_model(data_1, save='model1.h5')

    data_2 = data_loader('set_2')
    second_model(data_2, save='model2.h5')
    return None


if __name__ == '__main__':
    main()
