import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import float32 as ft_32
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Convolution2D
from tensorflow.keras.layers import Conv3D, AveragePooling3D, MaxPooling3D

from utils_ks import data_loader
from utils_ks import model_compilation, model_fit, set_compil_parameters, set_fit_parameters


def _dense_layer_definition(nb, activation=None):
    return Dense(units=nb, activation=activation)


def _compile_network(obj, optimizer, loss, metrics):
    return obj.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def _fit_network(obj, x_train, y_train, validation, epochs, batch_size, verbose=0):
    return obj.neural_network.fit(x=x_train, y=y_train, validation_data=validation,
                                  epochs=epochs, batch_size=batch_size, verbose=verbose)


def _evaluate_network(obj, x_test, y_test, batch_size=None, verbose=1):
    return obj.evaluate_network(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)


def _predict_network(obj, x_pred, batch_size=None, verbose=0):
    return obj.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)


def _save(obj, filename):
    return obj.save(filepath=filename)


def _summary(obj):
    return obj.summary()


# def fit_network(obj, )


class DN_Keras:
    def __init__(self, name=None):
        self.neural_network = Sequential(name=name)

    def add_input_layer(self, nb_lines, nb_cols=None):
        if nb_cols is None:
            input_def = Input(shape=(nb_lines,))
        else:
            input_def = Input(shape=(nb_lines, nb_cols))

        self.neural_network.add(input_def)
        return None

    def add_dense_layer(self, nb, activation='relu'):
        input_def = _dense_layer_definition(nb=nb, activation=activation)
        self.neural_network.add(input_def)
        return None

    def add_output_layer(self, nb, activation='softmax'):
        self.add_dense_layer(nb=nb, activation=activation)
        return None

    # def compile_network(self, optimizer, loss, metrics):
    #     _compile_network(obj=self.neural_network, optimizer=optimizer, loss=loss, metrics=metrics)
    #     return None
    #
    # def fit_network(self, x_train, y_train, validation, epochs, batch_size, verbose=0):
    #     # self.neural_network.fit(x=x_train,
    #     #                         y=y_train,
    #     #                         validation_data=validation,
    #     #                         epochs=epochs,
    #     #                         batch_size=batch_size,
    #     #                         verbose=verbose)
    #     _fit_network(obj=self.neural_network, x_train=x_train, y_train=y_train, validation=validation,
    #                  epochs=epochs, batch_size=batch_size, verbose=verbose)
    #     return None
    #
    # def evaluate_network(self, x_test, y_test, batch_size=None, verbose=1):
    #     # self.neural_network.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
    #     _evaluate_network(self.neural_network, x_test, y_test, batch_size=batch_size, verbose=verbose)
    #     return None
    #
    # def predict_network(self, x_pred, batch_size=None, verbose=0):
    #     # data_out = self.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)
    #     data_out = _predict_network(obj=self.neural_network, x_pred=x_pred, batch_size=None, verbose=0)
    #     return data_out
    #
    # def save(self, input_name):
    #     _save(self.neural_network, input_name)
    #     return None
    #
    # def summary(self):
    #     _summary(self.neural_network)
    #     return None
    #     <<<<<<<<<<      Parameters methods      >>>>>>>>>>
    def compile_network(self, optimizer, loss, metrics):
        _compile_network(obj=self.neural_network, optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def fit_network(self, x_train, y_train, validation, epochs, batch_size, verbose=0):
        # self.neural_network.fit(x=x_train,
        #                         y=y_train,
        #                         validation_data=validation,
        #                         epochs=epochs,
        #                         batch_size=batch_size,
        #                         verbose=verbose)
        _fit_network(obj=self.neural_network, x_train=x_train, y_train=y_train, validation=validation,
                     epochs=epochs, batch_size=batch_size, verbose=verbose)
        return None

    def evaluate_network(self, x_test, y_test, batch_size=None, verbose=1):
        # self.neural_network.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        _evaluate_network(self.neural_network, x_test, y_test, batch_size=batch_size, verbose=verbose)
        return None

    def predict_network(self, x_pred, batch_size=None, verbose=0):
        # data_out = self.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)
        data_out = _predict_network(obj=self.neural_network, x_pred=x_pred, batch_size=None, verbose=0)
        return data_out

    def save(self, input_name):
        _save(self.neural_network, input_name)
        return None

    def summary(self):
        _summary(self.neural_network)
        return None


# def second_model(data, fit=False, save=False):
#     net = MLP_KerasNetwork(name='2D_neural_network')
#
#     #     >>>>  Define your neural model using methods
#     #           --------------------------------------
#     net.add_input_layer(nb_lines=10, nb_cols=10)
#     net.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
#     net.add_pooling1D(typ='max', siz=2)
#     net.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
#     net.add_pooling1D(typ='max', siz=2)
#     net.add_flatten()
#     net.add_hidden_layer(nb=10, activation='softmax')
#     net.summary()
#     #     >>>>  Go to parametrize your model
#     #           ----------------------------
#     return net

class CNN_1D_Keras:

    def __init__(self, name=None):
        self.neural_network = Sequential(name=name)

    def add_input_layer(self, shape=None, batch_size=None, dtype=ft_32, name=None):
        input_def = Input(shape=shape, batch_size=batch_size, dtype=dtype, name=name)

        self.neural_network.add(input_def)
        return None

    def add_conv_layer(self, filters, kernel_size, strides=1, padding='valid', activation='relu'):
        input_def = Conv1D(filters=filters, kernel_size=kernel_size,
                           strides=strides, padding=padding, activation=activation)

        self.neural_network.add(input_def)
        return None

    def add_pooling_layer(self, pool_size=2, strides=None, padding='valid', dtype='max'):
        input_def = None

        if dtype is 'max':
            input_def = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)
        elif dtype is 'mean':
            input_def = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)

        self.neural_network(input_def)
        return None

    def add_flatten(self):
        input_def = Flatten()
        self.neural_network(input_def)
        return None

    def add_dense_layer(self, nb, activation='relu'):
        input_def = _dense_layer_definition(nb=nb, activation=activation)
        self.neural_network.add(input_def)
        return None

    def add_output_layer(self, nb, activation='softmax'):
        self.add_dense_layer(nb=nb, activation=activation)
        return None

    #     <<<<<<<<<<      Parameters methods      >>>>>>>>>>
    def compile_network(self, optimizer, loss, metrics):
        _compile_network(obj=self.neural_network, optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def fit_network(self, x_train, y_train, validation, epochs, batch_size, verbose=0):
        # self.neural_network.fit(x=x_train,
        #                         y=y_train,
        #                         validation_data=validation,
        #                         epochs=epochs,
        #                         batch_size=batch_size,
        #                         verbose=verbose)
        _fit_network(obj=self.neural_network, x_train=x_train, y_train=y_train, validation=validation,
                     epochs=epochs, batch_size=batch_size, verbose=verbose)
        return None

    def evaluate_network(self, x_test, y_test, batch_size=None, verbose=1):
        # self.neural_network.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        _evaluate_network(self.neural_network, x_test, y_test, batch_size=batch_size, verbose=verbose)
        return None

    def predict_network(self, x_pred, batch_size=None, verbose=0):
        # data_out = self.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)
        data_out = _predict_network(obj=self.neural_network, x_pred=x_pred, batch_size=None, verbose=0)
        return data_out

    def save(self, input_name):
        _save(self.neural_network, input_name)
        return None

    def summary(self):
        _summary(self.neural_network)
        return None


class CNN_2D_Keras:

    def __init__(self, name=None):
        self.neural_network = Sequential(name=name)

    def add_input_layer(self, shape=None, batch_size=None, dtype=None, name=None):
        input_def = Input(shape=shape, batch_size=batch_size, dtype=dtype, name=name)

        self.neural_network.add(input_def)
        return None

    def add_conv_layer(self, filters, kernel_size, strides=(1, 1), padding='valid', activation='relu'):
        input_def = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=strides, padding=padding, activation=activation)

        self.neural_network.add(input_def)
        return None

    def add_pooling_layer(self, pool_size=(2, 2), strides=None, padding='valid', dtype='max'):
        input_def = None

        if dtype is 'max':
            input_def = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        elif dtype is 'mean':
            input_def = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)

        self.neural_network(input_def)
        return None

    def add_flatten(self):
        input_def = Flatten()
        self.neural_network(input_def)
        return None

    def add_dense_layer(self, nb, activation='relu'):
        input_def = _dense_layer_definition(nb=nb, activation=activation)
        self.neural_network.add(input_def)
        return None

    def add_output_layer(self, nb, activation='softmax'):
        self.add_dense_layer(nb=nb, activation=activation)
        return None

    #     <<<<<<<<<<      Parameters methods      >>>>>>>>>>
    def compile_network(self, optimizer, loss, metrics):
        _compile_network(obj=self.neural_network, optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def fit_network(self, x_train, y_train, validation, epochs, batch_size, verbose=0):
        # self.neural_network.fit(x=x_train,
        #                         y=y_train,
        #                         validation_data=validation,
        #                         epochs=epochs,
        #                         batch_size=batch_size,
        #                         verbose=verbose)
        _fit_network(obj=self.neural_network, x_train=x_train, y_train=y_train, validation=validation,
                     epochs=epochs, batch_size=batch_size, verbose=verbose)
        return None

    def evaluate_network(self, x_test, y_test, batch_size=None, verbose=1):
        # self.neural_network.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        _evaluate_network(self.neural_network, x_test, y_test, batch_size=batch_size, verbose=verbose)
        return None

    def predict_network(self, x_pred, batch_size=None, verbose=0):
        # data_out = self.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)
        data_out = _predict_network(obj=self.neural_network, x_pred=x_pred, batch_size=None, verbose=0)
        return data_out

    def save(self, input_name):
        _save(self.neural_network, input_name)
        return None

    def summary(self):
        _summary(self.neural_network)
        return None


class CNN_3D_Keras:

    def __init__(self, name=None):
        self.neural_network = Sequential(name=name)


def model_definition():
    net = DN_Keras(name='1D_neural_network')

    #     >>>>  Define your neural model using methods
    #           --------------------------------------
    net.add_input_layer(nb_lines=100)
    net.add_dense_layer(nb=100, activation='relu')
    net.add_dense_layer(nb=50, activation='relu')
    net.add_output_layer(nb=10, activation='softmax')
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
    y_ctrl = '../data/data_set_1/y_ctrl.npy'

    return [x_train, y_train, x_ctrl, y_ctrl]


def data_filename2():
    x_train = '../data/data_set_1/x_train_2.npy'
    y_train = '../data/data_set_1/y_train.npy'

    x_ctrl = '../data/data_set_1/x_ctrl_2.npy'
    y_ctrl = '../data/data_set_1/y_ctrl.npy'

    return [x_train, y_train, x_ctrl, y_ctrl]


def main():
    filename_1 = data_filename()
    data_1 = data_loader(filename=filename_1, center=True, norm=True)
    model_1 = model_definition()

    compil_parameters, fit_parameters = model_settings()
    model_compilation(model=model_1, parameters=compil_parameters)
    model_fit(data=data_1, model=model_1, parameters=fit_parameters, fit=False)

    return None


if __name__ == '__main__':
    main()
