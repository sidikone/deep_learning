from tensorflow.keras import datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas import DataFrame, concat
from numpy import ndarray, array, concatenate


class DataSets:

    def __init__(self):
        self.__raw_data_tensor_flow = None
        self.__train_data_set = None
        self.__test_data_set = None
        self.__validation_data_set = None

        self.__feature_names = None
        self.__target_names = None

        self.__frame_column_names = None  # only for table
        self.__frame_data = None  # only for a table
        self.__frame_train = None
        self.__frame_test = None

        self.__sample_data = None
        self.__info = None
        self.__authentic_label = None  # only if label are not authentic

        self.__is_an_image = False
        self.__is_a_table = False
        self.__is_authentic_label = True

    def __set_authentic_fashion_mnist_label(self):
        self.__authentic_label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.__target_names = self.__authentic_label

    def __set_authentic_cifar10_label(self):
        self.__authentic_label = ["airplane", "automobile", "bird", "cat", "deer",
                                  "dog", "frog", "horse", "ship", "truck"]
        self.__target_names = self.__authentic_label

    def __set_authentic_iris_label(self):
        self.__authentic_label = list(self.__target_names)

    def __set_column_names_iris_label(self):
        self.__frame_column_names = list(self.__feature_names) + ["plant species (label)"]

    def __set_authentic_breast_cancer_label(self):
        self.__authentic_label = list(self.__target_names)

    def __set_column_names_breast_cancer_label(self):
        self.__frame_column_names = list(self.__feature_names) + ["type of cancer (label)"]

    def __set_authentic_wine_label(self):
        self.__authentic_label = list(self.__target_names)

    def __set_column_names_wine_label(self):
        self.__frame_column_names = list(self.__feature_names) + ["class of wine (label)"]

    def __set_column_names_boston_label(self):
        self.__frame_column_names = list(self.__feature_names) + ["MEDV (label)"]

    def __set_column_names_diabetes_label(self):
        self.__frame_column_names = list(self.__feature_names) + ["Glucose rate (label)"]

    @staticmethod
    def __redefine_label(actual, authentic_ref) -> ndarray:
        new_labels = []
        for elt in actual:
            new_labels.append(authentic_ref[elt])
        return array(new_labels)

    def __data_to_frame(self, data, label, columns):
        data_df = DataFrame(data, columns=self.__frame_column_names[:-1])
        label_df = DataFrame(label, columns=[self.__frame_column_names[-1]])
        final_df = concat([data_df, label_df], axis=1)
        return final_df

    def load_iris_dataset(self, train_size: float = .75) -> None:
        self.__is_a_table = True
        self.__is_authentic_label = False

        data_sets = load_iris()
        # data_keys = list(data_sets.keys())
        # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
        self.__raw_data_tensor_flow = tuple([data_sets.get('data'), data_sets.get('target')])
        data_train, data_test, target_train, target_test = train_test_split(data_sets.get('data'),
                                                                            data_sets.get('target'),
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

        self.__feature_names = data_sets.get('feature_names')
        self.__target_names = data_sets.get('target_names')
        self.__set_authentic_iris_label()
        self.__set_column_names_iris_label()

    def load_breast_cancer_dataset(self, train_size: float = .75) -> None:
        self.__is_a_table = True
        self.__is_authentic_label = False

        data_sets = load_breast_cancer()
        # data_keys = list(data_sets.keys())
        # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']

        self.__raw_data_tensor_flow = tuple([data_sets.get('data'), data_sets.get('target')])
        data_train, data_test, target_train, target_test = train_test_split(data_sets.get('data'),
                                                                            data_sets.get('target'),
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

        self.__feature_names = data_sets.get('feature_names')
        self.__target_names = data_sets.get('target_names')
        self.__set_authentic_breast_cancer_label()
        self.__set_column_names_breast_cancer_label()

    def load_diabetes_dataset(self, train_size: float = .75) -> None:
        self.__is_a_table = True

        data_sets = load_diabetes()
        # data_keys = list(data_sets.keys())
        # ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']

        self.__raw_data_tensor_flow = tuple([data_sets.get('data'), data_sets.get('target')])
        data_train, data_test, target_train, target_test = train_test_split(data_sets.get('data'),
                                                                            data_sets.get('target'),
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

        self.__feature_names = data_sets.get('feature_names')
        self.__target_names = array(["Glucose rate (label)"])
        self.__set_column_names_diabetes_label()

    def load_wine_dataset(self, train_size: float = .75) -> None:
        self.__is_a_table = True
        self.__is_authentic_label = False

        data_sets = load_wine()
        # data_keys = list(data_sets.keys())
        # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names']

        self.__raw_data_tensor_flow = tuple([data_sets.get('data'), data_sets.get('target')])
        data_train, data_test, target_train, target_test = train_test_split(data_sets.get('data'),
                                                                            data_sets.get('target'),
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

        self.__feature_names = data_sets.get('feature_names')
        self.__target_names = data_sets.get('target_names')
        self.__set_authentic_wine_label()
        self.__set_column_names_wine_label()

    def load_boston_housing_dataset(self, train_size: float = .75) -> None:
        self.__is_a_table = True

        data_sets = load_boston()
        # data_keys = list(data_sets.keys())
        # ['data', 'target', 'feature_names', 'DESCR', 'filename']

        self.__raw_data_tensor_flow = tuple([data_sets.get('data'), data_sets.get('target')])
        data_train, data_test, target_train, target_test = train_test_split(data_sets.get('data'),
                                                                            data_sets.get('target'),
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

        self.__feature_names = data_sets.get('feature_names')
        self.__target_names = array(["MEDV"])
        self.__set_column_names_boston_label()

        self.__frame_data = self.__data_to_frame(self.__raw_data_tensor_flow[0], self.__raw_data_tensor_flow[1],
                                                 columns=self.__frame_column_names)

        self.__frame_train = self.__data_to_frame(self.__train_data_set[0], self.__train_data_set[1],
                                                 columns=self.__frame_column_names)

        self.__frame_test = self.__data_to_frame(self.__test_data_set[0], self.__test_data_set[1],
                                                 columns=self.__frame_column_names)

    def load_mnist_dataset(self, train_size: float = .75) -> None:
        self.__is_an_image = True

        self.__raw_data_tensor_flow = datasets.mnist.load_data()
        (data_train_tensor, target_train_tensor), (data_test_tensor, target_test_tensor) = self.__raw_data_tensor_flow
        data_tensor = concatenate((data_train_tensor, data_test_tensor), axis=0)
        target_tensor = concatenate((target_train_tensor, target_test_tensor), axis=0)

        self.__raw_data_tensor_flow = tuple([data_tensor, target_tensor])
        data_train, data_test, target_train, target_test = train_test_split(data_tensor,
                                                                            target_tensor,
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])

    def load_fashion_mnist_dataset(self, train_size: float = .75) -> None:
        self.__is_an_image = True
        self.__is_authentic_label = False

        self.__raw_data_tensor_flow = datasets.fashion_mnist.load_data()
        (data_train_tensor, target_train_tensor), (data_test_tensor, target_test_tensor) = self.__raw_data_tensor_flow
        data_tensor = concatenate((data_train_tensor, data_test_tensor), axis=0)
        target_tensor = concatenate((target_train_tensor, target_test_tensor), axis=0)

        self.__raw_data_tensor_flow = tuple([data_tensor, target_tensor])
        data_train, data_test, target_train, target_test = train_test_split(data_tensor,
                                                                            target_tensor,
                                                                            train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])
        self.__set_authentic_fashion_mnist_label()

    def load_cifar10_dataset(self, train_size: float = .75) -> None:
        self.__is_an_image = True
        self.__is_authentic_label = False

        self.__raw_data_tensor_flow = datasets.cifar10.load_data()
        (data_train_tensor, target_train_tensor), (data_test_tensor, target_test_tensor) = self.__raw_data_tensor_flow
        data_tensor = concatenate((data_train_tensor, data_test_tensor), axis=0)
        target_tensor = concatenate((target_train_tensor, target_test_tensor), axis=0)

        self.__raw_data_tensor_flow = tuple([data_tensor, target_tensor])
        data_train, data_test, target_train, target_test = train_test_split(data_tensor,
                                                                            target_tensor,
                                                                            train_size=train_size)
        target_train = target_train.ravel()
        target_test = target_test.ravel()
        self.__train_data_set = tuple([data_train, target_train])
        self.__test_data_set = tuple([data_test, target_test])
        self.__set_authentic_cifar10_label()

    def get_raw_data(self):
        return self.__raw_data_tensor_flow

    def get_train_raw_data(self):
        return self.__train_data_set

    def get_test_raw_data(self):
        return self.__test_data_set

    def get_validation_raw_data(self):
        return self.__validation_data_set

    def get_data(self):
        return self.__frame_data

    def get_train_data(self):
        return self.__frame_train[self.__frame_train.columns[:-1]], self.__frame_train[[self.__frame_train.columns[-1]]]

    def get_test_data(self):
        return self.__frame_test[self.__frame_test.columns[:-1]], self.__frame_test[[self.__frame_test.columns[-1]]]

    def get_validation_data(self):
        return self.__validation_data_set

    def get_feature_names(self):
        return self.__feature_names

    def get_target_names(self):
        return self.__target_names

    def get_authentic_label(self):
        return self.__authentic_label

    def show_sample(self, start: int = 0, nb: int = 3, interpolation: str = 'none') -> None:
        local_data, local_label = self.__train_data_set
        local_data = local_data[start:nb]
        local_label = local_label[start:nb]

        if not self.__is_authentic_label:
            local_label = self.__redefine_label(actual=local_label, authentic_ref=self.__authentic_label)

        if self.__is_an_image:
            self.__display_images(data=local_data, label=local_label, interpol=interpolation, nb=nb)

        if self.__is_a_table:
            self.__display_table(data=local_data, label=local_label, nb=nb)

    def train_validation_split(self, train_size: float = .75) -> None:
        (data_train_init, target_train_init) = self.__train_data_set

        data_train, data_validation, target_train, target_validation = train_test_split(data_train_init,
                                                                                        target_train_init,
                                                                                        train_size=train_size)
        self.__train_data_set = tuple([data_train, target_train])
        self.__validation_data_set = tuple([data_validation, target_validation])

    def __display_images(self, data: ndarray, label: ndarray, interpol: str, nb: int = 0) -> None:
        max_col_size = 6
        if nb <= max_col_size:
            fig, axes = plt.subplots(1, nb)
            ax = axes.ravel()
            for ind in range(nb):
                ax[ind].imshow(data[ind], cmap="gray", interpolation=interpol)
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

    def __display_table(self, data: ndarray, label: ndarray, nb: int = 0) -> None:

        data_df = DataFrame(data, columns=self.__frame_column_names[:-1])
        label_df = DataFrame(label, columns=[self.__frame_column_names[-1]])
        final_df = concat([data_df, label_df], axis=1)
        print(final_df.head(nb))
