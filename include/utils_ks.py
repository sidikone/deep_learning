import numpy as np
from typing import List
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_mnist():
    return datasets.mnist.load_data()


def load_fashion_mnist():
    return datasets.fashion_mnist.load_data()


def load_boston_housing():
    return datasets.boston_housing.load_data()


def load_cifar10():
    return datasets.cifar10.load_data()


def load_cifar100():
    return datasets.cifar100.load_data()


def load_reuters():
    return datasets.reuters.load_data()


def load_imdb():
    return datasets.imdb.load_data()


def load_dataset(name='mnist'):
    data = {'mnist': load_mnist(),
            'fashion_mnist': load_fashion_mnist(),
            'boston_housing': load_boston_housing(),
            'cifar10': load_cifar10(),
            'cifar100': load_cifar100(),
            'reuters': load_reuters(),
            'imdb': load_imdb(),
            }
    return data.get(name)


def normalization(data_in):
    return data_in / np.max(data_in)


def mean_center(data_in):
    return data_in - np.mean(data_in)


def image_dataset_augmentation(data, rotation_range, width_shift_range, height_shift_range, zoom_range):
    gen = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range, zoom_range=zoom_range)
    return gen.fit(data)


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
    model.compile_network(optimizer=optim_in, loss=loss_in, metrics=metrics_in)
    return None


def model_compilation(model, parameters):
    parameters_compil(model=model, compil_vars=parameters)
    return None


def model_fit(data, model, parameters, fit=True, save=False):
    parameters_fit(model=model, data=data, fit_vars=parameters, fit_in=fit, save_in=save)
    return None
