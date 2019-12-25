import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import bagging
import grid_search
from downgrade import pca
from logistic_regression import LogisticRegression, accuracy, get_features, \
    normalize

dirname = os.path.dirname(__file__)

df_train = pd.read_csv(os.path.join(dirname, 'data/train.csv'))
train_data = df_train.values
df_test = pd.read_csv(os.path.join(dirname, 'data/test.csv'))
test_data = df_test.values
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_train = np.stack([np.ones_like(x_train[:, 0]), *x_train.T], axis=1)
x_test, y_test = test_data[:, :-1], test_data[:, -1]
x_test = np.stack([np.ones_like(x_test[:, 0]), *x_test.T], axis=1)
power = 5

x_train1 = get_features(x_train, power)
x_train1 = normalize(x_train1)
v = pca(x_train1)
x_train1 = x_train1 @ v
#
x_test1 = get_features(x_test, power)
x_test1 = normalize(x_test1)
x_test1 = x_test1 @ v

if __name__ == "__main__":
    accuracies = {}
    # print('Logistic regression')
    #
    # model = LogisticRegression()
    # model.fit(x_train1, y_train)
    # y = model.predict(x_train1)
    # # plt.scatter(x_train1[:, 0], x_train1[:, 1], c=y)
    # # plt.show()
    # train_accuracy = accuracy(y_train, y) * 100
    # print(f"Train accuracy: {train_accuracy}")
    # y = model.predict(x_test1)
    # # plt.scatter(x_test1[:, 0], x_test1[:, 1], c=y)
    # # plt.show()
    # test_accuracy = accuracy(y_test, y) * 100
    # accuracies['LR'] = test_accuracy
    # print(f"Test accuracy: {test_accuracy}")

    print('Bagging based on logistic regression')
    base_model = LogisticRegression()
    model = bagging.BaggingClasifier(base_model=base_model, sample_size=70,
                                     iters=100)
    model.fit(x_train1, y_train)
    y = model.predict(x_train1)
    plt.scatter(x_train1[:, 0], x_train1[:, 1], c=y)
    plt.show()
    train_accuracy = accuracy(y_train, y) * 100
    print(f"Train accuracy: {train_accuracy}")
    y = model.predict(x_test1)
    plt.scatter(x_test1[:, 0], x_test1[:, 1], c=y)
    plt.show()
    test_accuracy = accuracy(y_test, y) * 100
    accuracies['Bagging LR'] = test_accuracy

    print(f"Test accuracy: {test_accuracy}")
    #
    # print('KNN')
    # model = KNeighborsClassifier(n_neighbors=19, p=2)
    # model.fit(x_train1, y_train)
    # y = model.predict(x_train1)
    # print(f"Train accuracy: {accuracy(y_train, y) * 100}")
    # y = model.predict(x_test1)
    # test_accuracy = accuracy(y_test, y) * 100
    # accuracies['KNN'] = test_accuracy
    # print(f"Test accuracy: {test_accuracy}")
    #
    # print('Bagging based on KNN')
    # base_model = KNeighborsClassifier(n_neighbors=5)
    # model = bagging.BaggingClasifier(base_model=base_model, sample_size=70,
    #                                  iters=100)
    # model.fit(x_train1, y_train)
    # y = model.predict(x_train1)
    # train_accuracy = accuracy(y_train, y) * 100
    # print(f"Train accuracy: {train_accuracy}")
    # y = model.predict(x_test1)
    # test_accuracy = accuracy(y_test, y) * 100
    # accuracies['Bagging KNN'] = test_accuracy
    # print(f"Test accuracy: {test_accuracy}")
    #
    # plt.bar(np.arange(len(accuracies)), accuracies.values())
    # plt.xticks(np.arange(len(accuracies)), accuracies.keys())
    # plt.ylim(50, 100)
    #
    # plt.show()

    # print('Logistic regression grid search')
    # model = LogisticRegression()
    # rates, best_rate = grid_search.search(model, 'learning_rate',
    #                                       [4, 3, 0.9, 0.3, 3e-2, 3e-3],
    #                                       x_train1, y_train, x_test1, y_test)
    #
    # plt.plot(list(rates), list(rates.values()))
    # plt.scatter([best_rate], [rates[best_rate]], color='red')
    # plt.annotate(str(best_rate), xy=(best_rate, rates[best_rate]))
    # plt.xlabel('Learning rate')
    # plt.ylabel('Test accuracy')
    # plt.show()

    # print('KNN grid search')
    # model = KNeighborsClassifier()
    # rates, best_rate = grid_search.search(model, 'n_neighbors',
    #                                       range(1, 50),
    #                                       x_train1, y_train, x_test1, y_test)
    #
    # plt.plot(list(rates), list(rates.values()))
    # plt.scatter([best_rate], [rates[best_rate]], color='red')
    # plt.annotate(str(best_rate), xy=(best_rate, rates[best_rate]))
    # plt.xlabel('Number of neighbors')
    # plt.ylabel('Test accuracy')
    # print(rates[best_rate])
    # plt.show()

    # print('Bagging grid search')
    # base_model = LogisticRegression()
    # model = bagging.BaggingClasifier(base_model=base_model,
    #                                  iters=150)
    # rates, best_rate = grid_search.search(model, 'sample_size',
    #                                       range(10, 100, 10),
    #                                       x_train1, y_train, x_test1, y_test)
    #
    # plt.plot(list(rates), list(rates.values()))
    # plt.scatter([best_rate], [rates[best_rate]], color='red')
    # plt.annotate(str(best_rate), xy=(best_rate, rates[best_rate]))
    # plt.xlabel('Size of sample')
    # plt.ylabel('Test accuracy')
    # print(rates[best_rate])
    # plt.show()
