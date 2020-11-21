import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np

print(tf.__version__)


class TensflowNetwork:

    def __init__(self, name=None):
        self.network_name = name
        self.nb_unnamed_layer = 0

        self.layer_names = []
        self.no_layer = True

    @property
    def get_layer_names(self):
        return self.layer_names

    @staticmethod
    def add_input(nb_lines, nb_cols=None, typ=tf.float32, name=None):

        typ_in = typ
        name_in = name
        if nb_cols is None:
            shape_in = (nb_lines,)
        else:
            shape_in = (nb_lines, nb_cols)

        input_def = v1.placeholder(dtyp=typ_in, shape=shape_in, name=name_in)

        return input_def

    def _set_name_layer(self, name_in):
        # local_name = ''
        if name_in is 'layer' and self.no_layer:
            local_name = name_in

            self.nb_unnamed_layer += 1
            self.no_layer = False

        elif name_in is 'layer' and self.nb_unnamed_layer:
            local_name = name_in + '_' + str(self.nb_unnamed_layer)
            self.nb_unnamed_layer += 1

        else:
            local_name = name_in
        self.layer_names.append(local_name)
        return local_name

    def _add_neuron_layer(self, previous_layer, nb_neurons, name='layer', activation=None):

        # 0 -> instances & 1 -> characteristics
        # instances correspond to the batch_size
        # characteristics correspond to the number of output of the previous layer

        local_name = self._set_name_layer(name_in=name)
        with v1.name_scope(local_name):
            nb_inputs = (previous_layer.get_shape()[1])
            std_dev = 2 / np.sqrt(nb_inputs + nb_neurons)
            init = v1.truncated_normal((nb_inputs, nb_neurons), stddev=std_dev)

            W = v1.Variable(init, name='kernel')
            b = v1.Variable(v1.zeros([nb_neurons]), name='bias')
            z = v1.matmul(previous_layer, W) + b

            if activation:
                return activation(z)
            else:
                return z

    # def add_dense(self,  nb, activation=None):
    #



if __name__ == '__main__':
    test = TensflowNetwork()
