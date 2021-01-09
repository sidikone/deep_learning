from include.data_load import DataSets


def main() -> None:
    datasets_obj = DataSets()
    datasets_obj.load_diabetes_dataset()

    train_data, train_label = datasets_obj.get_train_data()
    test_data, test_label = datasets_obj.get_test_data()
    datasets_obj.show_sample(nb=5)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
