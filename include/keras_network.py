import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D

from utils_ml import mean_center, normalization, data_loader
from utils_ml import model_compilation, model_fit, set_compil_parameters, set_fit_parameters

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

    def evaluate_network(self, x_test, y_test, batch_size=None, verbose=1):
        self.neural_network.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        return None

    def predict_network(self, x_pred, batch_size=None, verbose=0):
        data_out = self.neural_network.predict(x=x_pred, batch_size=batch_size, verbose=verbose)
        return data_out

    def save(self, input_name):
        self.neural_network.save(filepath=input_name)
        return None

    def summary(self):
        self.neural_network.summary()
        return None


def second_model(data, fit=False, save=False):
    net = NeuralNetwork(name='2D_neural_network')

    #     >>>>  Define your neural model using methods
    #           --------------------------------------
    net.add_input(nb_lines=10, nb_cols=10)
    net.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
    net.add_pooling1D(typ='max', siz=2)
    net.add_conv1D(filter_nb=16, filter_siz=3, padding_type='same', activation='relu')
    net.add_pooling1D(typ='max', siz=2)
    net.add_flatten()
    net.add_dense(nb=10, activation='softmax')
    net.summary()
    #     >>>>  Go to parametrize your model
    #           ----------------------------
    return net


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
