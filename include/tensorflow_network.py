import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1

from utils_tf import activation_fn

tf.compat.v1.disable_eager_execution()


print(tf.__version__)


class DNNetwork_tf:
    def __init__(self, name=None):
        self.network_name = name
        self.nb_unnamed_layer = 0
        self.index_previous = 0

        self.layer_names = []
        self.previous_network_state = None
        self.actual_network_state = None
        self.neural_network = None
        self.no_layer = True

    @property
    def get_layer_names(self):
        return self.layer_names

    def _set_name_layer(self, name_in):
        # local_name = ''
        if name_in is "layer" and self.no_layer:
            local_name = name_in

            self.nb_unnamed_layer += 1
            self.no_layer = False

        elif name_in is "layer" and self.nb_unnamed_layer:
            local_name = name_in + "_" + str(self.nb_unnamed_layer)
            self.nb_unnamed_layer += 1

        else:
            local_name = name_in
        self.layer_names.append(local_name)
        return local_name

    def _add_neuron_layer(
        self, previous_layer, nb_neurons, name="layer", activation=None
    ):

        # 0 -> instances & 1 -> characteristics
        # instances correspond to the batch_size
        # characteristics correspond to the number of output of the previous layer

        local_name = self._set_name_layer(name_in=name)
        with v1.name_scope(local_name):
            nb_inputs = previous_layer.get_shape()[1]
            std_dev = 2 / np.sqrt(nb_inputs + nb_neurons)
            init = v1.truncated_normal((nb_inputs, nb_neurons), stddev=std_dev)

            W = v1.Variable(init, name="kernel")
            b = v1.Variable(v1.zeros([nb_neurons]), name="bias")
            z = v1.matmul(previous_layer, W) + b

            if activation:
                return activation(z)
            else:
                return z

    def add_input(self, nb_lines, nb_cols=None, typ=tf.float32, name="input"):

        typ_in = typ
        name_in = name
        if nb_cols is None:
            shape_in = (None, nb_lines)
        else:
            shape_in = (nb_lines, nb_cols)

        input_def = v1.placeholder(dtype=typ_in, shape=shape_in, name=name_in)
        self.previous_network_state = input_def
        # self.previous_network_state = deepcopy(input_def)

        return None

    def update_previous_network_state_for_the_next_sate(self):

        self.previous_network_state = self.actual_network_state
        return None

    def add_dense(self, nb, activation=None, name=None):

        local_name = self._set_name_layer(name_in=name)
        with v1.name_scope(name=self.network_name):
            self.actual_network_state = v1.layers.dense(
                inputs=self.previous_network_state,
                units=nb,
                name=local_name,
                activation=activation,
            )
        self.update_previous_network_state_for_the_next_sate()
        return None

    def add_output(self, nb, activation=None, name="outputs"):
        local_name = name
        with v1.name_scope(name=self.network_name):
            self.actual_network_state = v1.layers.dense(
                inputs=self.previous_network_state,
                units=nb,
                name=local_name,
                activation=activation,
            )
        self.neural_network = self.actual_network_state
        return None

    # def summary(self):

    # model_vars = v1.trainable_variables()
    # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    #    return None


def model_definition():
    net = DNNetwork_tf(name="1D_neural_network")

    #     >>>>  Define your neural model using methods
    #           --------------------------------------
    net.add_input(nb_lines=100)
    net.add_dense(nb=100, activation=activation_fn())
    #    net.add_dense(nb=50, activation='relu')
    #    net.add_output(nb=10, activation='softmax')

    #     >>>>  Go to parametrize your model
    #           ----------------------------
    return net


def main():
    model_1 = model_definition()
    return None


if __name__ == "__main__":
    main()
