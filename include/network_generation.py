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
from utils import mean_center, normalization


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


def data_loader(filename, norm=True, center=True) -> List:
    x_train_1, x_ctrl_1, y_train_1, y_ctrl_1 = filename
    x_train = np.load(x_train_1)
    x_ctrl = np.load(x_ctrl_1)

    if norm:
        x_train = normalization(x_train)
        x_ctrl = normalization(x_ctrl)

    elif center:
        x_train = mean_center(x_train)
        x_ctrl = mean_center(x_ctrl)

    y_train = np.load(y_train_1)
    y_train = to_categorical(y_train)
    y_ctrl = np.load(y_ctrl_1)
    y_ctrl = to_categorical(y_ctrl)

    return [x_train, y_train, x_ctrl, y_ctrl]


def set_fit_parameters(batch_size, epochs, verbose=2):
    return [batch_size, epochs, verbose]


def parameters_fit(model, data, fit_vars, fit_in, save_in):
    if fit_in:
        [x_train, y_train, x_ctrl, y_ctrl] = data
        [batch_size, epochs, verbose] = fit_vars
        model.fit_network(x_train, y_train,
                          validation=(x_ctrl, y_ctrl),
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose)
    if save_in:
        model.save(save_in)
    return None


def set_compil_parameters(optim, loss, metrics):
    return [optim, loss, metrics]


def parameters_compil(model, compil_vars):
    [optim_in, loss_in, metrics_in] = compil_vars
    model.compile_network(optim=optim_in, loss=loss_in, metrics=metrics_in)
    return None


def first_model(data, fit_pars, compil_pars, fit=False, save=False):
    net = NeuralNetwork(name='1D_neural_network')

    #     >>>>  Define your neural model
    #           ------------------------
    net.add_input(nb_lines=100)
    net.add_dense(nb=100, activation='relu')
    net.add_dense(nb=50, activation='relu')
    net.add_dense(nb=10, activation='softmax')
    net.summary()

    #     <<<<  Compile and fit don't modify
    #           ----------------------------
    parameters_compil(model=net, compil_vars=compil_pars)
    parameters_fit(model=net, data=data, fit_vars=fit_pars, fit_in=fit, save_in=save)
    return None


def model_compilation(model, parameters):
    parameters_compil(model=model, compil_vars=parameters)
    return None


def model_fit(data, model, parameters, fit=True, save=False):
    parameters_fit(model=model, data=data, fit_vars=parameters, fit_in=fit, save_in=save)
    return None


def second_model(data, fit=False, save=False):
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
    net2.compile_network(optim=opt_net)

    if fit:
        net2.fit_network(x_train, y_train, validation=(x_ctrl, y_ctrl), batch_size=128, epochs=500, verbose=2)

    if save is not False:
        net2.save(save)
    return None


def model_definition():
    net = NeuralNetwork(name='1D_neural_network')

    #     >>>>  Define your neural model using methods
    #           --------------------------------------
    net.add_input(nb_lines=100)
    net.add_dense(nb=100, activation='relu')
    net.add_dense(nb=50, activation='relu')
    net.add_dense(nb=10, activation='softmax')
    net.summary()

    #     >>>>  Go to parametrize your model
    #           ----------------------------
    return net


def model_settings():
    opt_net = optimizers.Adam(learning_rate=1e-2)
    compil = set_compil_parameters(optim=opt_net, loss='categorical_crossentropy', metrics=['accuracy'])
    fit = set_fit_parameters(batch_size=128, epochs=500)

    return compil, fit


def data_filename():
    x_train = '../data/data_set_1/x_train_1.npy'
    y_train = '../data/data_set_1/y_train.npy'

    x_ctrl = '../data/data_set_1/x_ctrl_1.npy'
    y_ctrl = np.load('../data/data_set_1/y_ctrl.npy')

    return [x_train, y_train, x_ctrl, y_ctrl]


def main():
    filename_1 = data_filename()
    data_1 = data_loader(filename=filename_1, center=True, norm=True)

    model_1 = model_definition()

    compil_parameters, fit_parameters = model_settings()
    model_compilation(model=model_1, parameters=compil_parameters)
    model_fit(data=data_1, model=model_1, parameters=fit_parameters, fit=False)

    # first_model(data=data_1, fit_pars=parameters_1, compil_pars=compil_values_1)
    # print('\n\n')
    # data_2 = data_loader('set_2')
    # second_model(data_2)
    return None


if __name__ == '__main__':
    main()
