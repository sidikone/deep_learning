import tensorflow as tf
import tensorflow.compat.v1 as v1


def activation_fn(name='relu'):
    my_config = {'relu': v1.nn.relu,
                 'selu': v1.nn.selu,
                 'elu': v1.nn.elu,
                 'softmax': v1.nn.softmax,
                 'sigmoid': v1.nn.sigmoid,
                 'tanh': v1.nn.tanh}
    return my_config.get(name)
