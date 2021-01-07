from tensorflow.keras import datasets
from matplotlib import pyplot as plt
from numpy import ndarray, zeros, ones, uint8, array


class DataSets:

    def __init__(self):
        self.__raw_data = None
        self.__train_data_set = None
        self.__test_data_set = None
        self.__frame_data = None
        self.__sample_data = None
        self.__info = None
        self.__authentic_label = None

        self.__is_an_image = False
        self.__is_a_table = False
        self.__is_authentic_label = True

    def __authentic_fashion_mnist_label(self):
        self.__authentic_label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def __redefine_label(self, actual, authentic_ref) -> ndarray:
        new_labels = []
        for elt in actual:
            new_labels.append(authentic_ref[elt])
        return array(new_labels)

    def load_mnist(self) -> None:
        self.__is_an_image = True
        self.__raw_data = datasets.mnist.load_data()
        self.__train_data_set, self.__test_data_set = self.__raw_data
        return None

    def load_fashion_mnist(self) -> None:
        self.__is_an_image = True
        self.__is_authentic_label = False

        self.__raw_data = datasets.fashion_mnist.load_data()
        self.__train_data_set, self.__test_data_set = self.__raw_data
        self.__authentic_fashion_mnist_label()

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
        local_data = local_data[start:nb]
        local_label = local_label[start:nb]

        if not self.__is_authentic_label:
            local_label = self.__redefine_label(actual=local_label, authentic_ref=self.__authentic_label)

        if self.__is_an_image:
            self.__display_images(data=local_data, label=local_label, nb=nb)

    def __display_images(self, data: ndarray, label: ndarray, nb: int = 0) -> None:
        max_col_size = 6
        if nb <= max_col_size:
            fig, axes = plt.subplots(1, nb)
            ax = axes.ravel()
            for ind in range(nb):
                ax[ind].imshow(data[ind], cmap="gray")
                ax[ind].set_title(label[ind], color="r")
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

                        if self.__is_authentic_label:
                            axes[line, col].set_ylabel('{0} {1}'.format(label[data_ind], " " * 5), fontsize=10,
                                                       rotation=0,
                                                       color="r")
                        else:
                            axes[line, col].set_ylabel('{0} {1}'.format(label[data_ind], " " * 5), fontsize=10,
                                                       color="r")
                        axes[line, col].axes.xaxis.set_ticks([])
                        axes[line, col].axes.yaxis.set_ticks([])
                        data_ind += 1

                    else:
                        axes[line, col].set_title('{}'.format(" "))
                        axes[line, col].axis('off')
            plt.show()
