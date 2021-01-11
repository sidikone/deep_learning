from include.data_load import DataSets
from include.regression import Regression
import numpy as np


def main() -> None:
    datasets_obj = DataSets()
    datasets_obj.load_boston_housing_dataset()

    train_data, train_target = datasets_obj.get_train_data()
    test_data, test_target = datasets_obj.get_test_data()

    reg = Regression("linear")
    reg.fit(train_data, train_target)
    train_score = reg.get_score(train_data, train_target)
    test_score = reg.get_score(test_data, test_target)

    predict = reg.predict(test_data)


if __name__ == '__main__':
    main()
