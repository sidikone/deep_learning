from tensorflow.keras import datasets
from matplotlib import pyplot as plt
from numpy import ndarray, zeros, ones, uint8


class DataSets:

    def __init__(self):
        self.__raw_data = None
        self.__train_data_set = None
        self.__test_data_set = None
        self.__frame_data = None
        self.__sample_data = None
        self.__info = None

        self.__is_an_image = False
        self.__is_a_table = False

    def load_mnist(self):
        self.__is_an_image = True
        self.__raw_data = datasets.mnist.load_data()
        self.__train_data_set, self.__test_data_set = self.__raw_data
        return None

    def load_fashion_mnist(self):
        self.__raw_data = datasets.fashion_mnist.load_data()
        self.__train_data_set, self.__test_data_set = self.__raw_data
        return self.__raw_data

    def load_cifar10(self):
        self.__raw_data = datasets.cifar10.load_data()
        return self.__raw_data

    def load_cifar100(self):
        self.__raw_data = datasets.cifar100.load_data()
        return self.__raw_data

    def get_raw_data(self):
        return self.__raw_data

    def get_train_data(self):
        return self.__train_data_set

    def get_test_data(self):
        return self.__test_data_set

    def show_sample(self, start: int = 0, nb: int = 3):
        local_data, local_label = self.__train_data_set
        local_data = local_data[start:]
        local_label = local_label[start:]

        if self.__is_an_image:
            self.__display_images(data=local_data, label=local_label, nb=nb)

    @staticmethod
    def __display_images(data: ndarray, label: ndarray, nb: int = 0) -> None:
        max_col_size = 6
        if nb < max_col_size:
            fig, axes = plt.subplots(1, nb)
            ax = axes.ravel()
            for ind in range(nb):
                ax[ind].imshow(data[ind], cmap="gray")
                ax[ind].set_title('{}'.format(label[ind]))
                ax[ind].axis('off')
            plt.show()

        else:
            nb_line = nb // max_col_size
            nb_elt_last_line = nb % max_col_size

            if nb_elt_last_line is 0:
                max_line_size = nb_line
            else:
                max_line_size = nb_line + 1

            fig, axes = plt.subplots(max_line_size, max_col_size)
            data_ind = 0
            for line in range(max_line_size):
                for col in range(max_col_size):

                    if data_ind < nb:
                        axes[line, col].imshow(data[data_ind], cmap="gray")
                        axes[line, col].set_title('{}'.format(label[data_ind]))
                        axes[line, col].axis('off')
                        data_ind += 1

                    else:
                        axes[line, col].set_title('{}'.format(" "))
                        axes[line, col].axis('off')


            plt.show()

