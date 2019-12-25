import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import bagging
from downgrade import pca
from logistic_regression import LogisticRegression, accuracy, get_features, normalize

dirname = os.path.dirname(__file__)

df_train = pd.read_csv(os.path.join(dirname, 'data/train.csv'))
train_data = df_train.values
df_test = pd.read_csv(os.path.join(dirname, 'data/test.csv'))
test_data = df_test.values
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_train = np.stack([np.ones_like(x_train[:, 0]), *x_train.T], axis=1)
x_test, y_test = test_data[:, :-1], test_data[:, -1]
x_test = np.stack([np.ones_like(x_test[:, 0]), *x_test.T], axis=1)
power = 4

x_train1 = get_features(x_train, power)
x_train1 = normalize(x_train1)
v = pca(x_train1)
x_train1 = x_train1 @ v
#
x_test1 = get_features(x_test, power)
x_test1 = normalize(x_test1)
x_test1 = x_test1 @ v


if __name__ == "__main__":
    base_model = LogisticRegression()
    model = bagging.BaggingClasifier(base_model=base_model, sample_size=70, iters=150)
    model.fit(x_train1, y_train)
    y = model.predict(x_train1)
    train_accuracy = accuracy(y_train, y) * 100
    print(f"Train accuracy: {train_accuracy}")
    y = model.predict(x_test1)
    test_accuracy = accuracy(y_test, y) * 100
    print(f"Test accuracy: {test_accuracy}")

    model = LogisticRegression()
    model.fit(x_train1, y_train)
    y = model.predict(x_train1)
    train_accuracy = accuracy(y_train, y) * 100
    print(f"Train accuracy: {train_accuracy}")
    y = model.predict(x_test1)
    test_accuracy = accuracy(y_test, y) * 100
    print(f"Test accuracy: {test_accuracy}")

    # model = KNeighborsClassifier(n_neighbors=3, p=1)
    # model.fit(x_train1, y_train)
    # y = model.predict(x_train1)
    # print(f"Train accuracy: {accuracy(y_train, y) * 100}")
    # y = model.predict(x_test1)
    # print(f"Test accuracy: {accuracy(y_test, y) * 100}")


# Train accuracy: 88.42975206611571
# Test accuracy: 86.88524590163934