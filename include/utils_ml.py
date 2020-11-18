import numpy as np
from typing import List
from tensorflow.keras.utils import to_categorical


def normalization(data_in):
    return data_in / np.max(data_in)


def mean_center(data_in):
    return data_in - np.mean(data_in)


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


def model_compilation(model, parameters):
    parameters_compil(model=model, compil_vars=parameters)
    return None


def model_fit(data, model, parameters, fit=True, save=False):
    parameters_fit(model=model, data=data, fit_vars=parameters, fit_in=fit, save_in=save)
    return None
