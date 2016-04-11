from openml.apiconnector import APIConnector
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss, roc_auc_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold, Bootstrap, ShuffleSplit
from sklearn import cross_validation

# #connect to openml api
# apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
# connector = APIConnector(apikey=apikey)
# dataset = connector.download_dataset(44)
# X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
# X = pd.DataFrame(X, columns=attribute_names)
# print X[:2]

def exercise_1():
    #connect to openml api
    apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
    connector = APIConnector(apikey=apikey)
    dataset = connector.download_dataset(44)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

    error = []
    lst = [int(math.pow(2, i)) for i in range(0, 8)]
    # lst_2 = [i for i in range(1, 200)]
    #train the classifier
    clf = RandomForestClassifier(oob_score=True,
                                   max_features="auto",
                                   random_state=0)
    #loop estimator parameter
    for i in lst:
        clf.set_params(n_estimators=i)
        clf.fit(X, y)
        error.append(1 - clf.oob_score_)
    #plot
    plt.style.use('ggplot')
    plt.scatter(lst, error)
    plt.xticks(lst)
    plt.show()

def exercise_2():
    #connect to openml api
    apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
    connector = APIConnector(apikey=apikey)
    dataset = connector.download_dataset(44)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

    kf = cross_validation.KFold(len(X), n_folds=10, shuffle=False, random_state=0)
    error = []
    error_mean = []
    lst = [int(math.pow(2, i)) for i in range(0, 8)]
    clf = RandomForestClassifier(oob_score=True,
                                   max_features="auto",
                                   random_state=0)
    for i in lst:
        error_mean = []
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)
            error_mean.append( zero_one_loss(y_test, clf.predict(X_test)) )
        error.append( np.array(error_mean).mean() )
    #plot
    plt.style.use('ggplot')
    plt.scatter(lst, error)
    plt.xticks(lst)
    plt.show()

def exercise_3():
    #connect to openml api
    apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
    connector = APIConnector(apikey=apikey)
    dataset = connector.download_dataset(44)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

    kf = cross_validation.ShuffleSplit(len(X),n_iter=100, test_size=0.1, train_size=0.9, random_state=0)
    error = []
    error_mean = []

    clf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                   max_features="auto",
                                   random_state=0)

    error_mean = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        error_mean.append( roc_auc_score(y_test, clf.predict(X_test)) )
    error.append( np.array(error_mean).mean() )

    print error

    #plot
    # plt.style.use('ggplot')
    # plt.scatter(lst, error)
    # plt.xticks(lst)
    # plt.show()


if __name__ == "__main__":
    exercise_3()
