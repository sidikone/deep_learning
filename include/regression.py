from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso


class Regression:
    def __init__(self, model_type: str = "linear"):
        self.__model = None
        self.__coeff = None
        self.__const = None
        self.__score = None
        self.__result = None

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

    def fit(self, train_data, train_label):
        self.__model.fit(train_data, train_label)
        self.__coeff = self.__model.coef_
        self.__const = self.__model.intercept_

    def predict(self, data):
        self.__result = self.__model.predict(data)
        return self.__result

    def get_parameters(self):
        return self.__coeff, self.__const

    def get_score(self, data, target):
        self.__score = self.__model.score(data, target)
        return self.__score