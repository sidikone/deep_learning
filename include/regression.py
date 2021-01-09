from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

# from tensorflow.estimator import LinearRegressor
# import tensorflow as tf
# import os
# import numpy as np
# import pandas as pd
# import mglearn.datasets as mg_datasets
# from sklearn.model_selection import train_test_split
# from utils_ks import load_boston_housing
#
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# numeric_column = tf.feature_column.numeric_column
# categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list


class Regression:
    def __init__(self, model_type: str = "linear"):
        self.__model = None
        self.__initialize_model(typ=model_type)

    def __initialize_model(self, typ: str):
        if typ is "linear":
            self.__model = LinearRegression()

        if typ is "ridge":
            self.__model = Ridge()

        if typ is "lasso":
            self.__model = Lasso()

        if typ is "logistic":
            self.__model = LogisticRegression()

    def set_ridge_parameters(self, alpha=1):
        self.__model = Ridge(alpha=alpha)

    def set_lasso_parameters(self, alpha=1, max_inter=1000, normalize=False):
        self.__model = Lasso(alpha=alpha, max_iter=max_inter, normalize=normalize)

    def set_logistic_parameters(self, balance=1):
        self.__model = LogisticRegression(C=balance)

    def fit(self, train_data,  target_data):
        self.__model.fit(train_data,  target_data)

    def predict(self, data):
        self.__model.predict(data)



# class LogisticRegression:
#     def __init__(self):
#         print("Hello world")


# def train_input_fn():
#     features = {"area": [1000, 2000, 4000, 1000, 2000, 4000],
#                 "type": ["bungalow", "bungalow", "bungalow",
#                          "apartment", "apartment", "apartment"]}
#     labels = [500, 1000, 1500, 700, 1300, 1900]
#     return features, labels
#
#
# def predict_input_fn():
#     features = {"area": [1500, 1800],
#                 "type": ["bungalow", "apartment"]}
#     return features


def main() -> None:
    reg = Regression()

    # data_train, data_test, data_features = load_boston_housing()
    # x_train, y_train = data_train
    # x_test, y_test = data_test
    # x_features, y_features = data_features
    #
    # x_train_df = pd.DataFrame(x_train, columns=x_features)
    # x_test_df = pd.DataFrame(x_test, columns=x_features)
    #
    # y_train_df = pd.DataFrame(y_train, columns=y_features)
    # y_test_df = pd.DataFrame(y_test, columns=y_features)
    #
    # print(x_train_df.head(3))
    #
    # X, y = mg_datasets.make_wave(n_samples=100)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# def main_1():
#     feature_cols = [numeric_column("area"),
#                     categorical_column_with_vocabulary_list("type", ["bungalow", "apartment"])]
#     model = LinearRegressor(feature_cols)
#     model.train(train_input_fn, steps=180)
#
#     predictions = model.predict(predict_input_fn)
#     # print(next(predictions))
#     # print(next(predictions))
#     return None


if __name__ == '__main__':
    main()
