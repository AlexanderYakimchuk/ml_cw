from copy import deepcopy

from logistic_regression import accuracy


def search(model, param_name, range_, x_train, y_train,  x_test, y_test):
    res = {}
    for param in range_:
        model = deepcopy(model)
        setattr(model, param_name, param)
        model.fit(x_train, y_train)
        a = accuracy(model.predict(x_test), y_test)
        res[param] = a
    sorted_res = list(sorted(deepcopy(res), key=lambda x: -res[x]))
    return res, sorted_res[0]
