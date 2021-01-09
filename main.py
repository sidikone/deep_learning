from include.data_load import DataSets


def main() -> None:
    datasets_obj = DataSets()
    datasets_obj.load_cifar10_dataset()
    datasets_obj.train_validation_split()

    train_data, train_label = datasets_obj.get_train_data()
    validation_data, validation_label = datasets_obj.get_validation_data()
    test_data, test_label = datasets_obj.get_test_data()
    print(train_data.shape)
    print(validation_data.shape)
    print(test_data.shape)

    datasets_obj.show_sample(nb=10)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
