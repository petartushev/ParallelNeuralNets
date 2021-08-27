import time
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time
import concurrent.futures
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import datetime
from collections import Counter


def train_neural_net(data):
    # print(f"Job started at {datetime.datetime.now()}")
    clf = MLPClassifier(30, activation="tanh", learning_rate_init=0.002, max_iter=200, random_state=0)
    clf.fit(data[0], data[1])
    return clf


if __name__ == '__main__':
    # print(time.ctime())
    raw_data = pd.read_csv("D:\\pidp dataset\\airline_passenger_satisfaction.csv")
    raw_data = raw_data.dropna()
    # print(raw_data.shape)
    array = np.asarray(raw_data)

    encoder = OrdinalEncoder()
    data = encoder.fit_transform(array)
    data = pd.DataFrame(data)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    x = np.asarray(x)
    y = np.asarray(y)

    x_test = x[int(len(x) * 0.9):]
    y_test = y[int(len(y) * 0.9):]

    start = time.perf_counter()

    clf = train_neural_net([x, y])

    acc = 0

    for x, y in zip(x_test, y_test):
        predict = clf.predict([x])
        if predict == y:
            acc += 1

    print(f"Accuracy of the parallel neural nets is {acc / len(y_test)}")

    finish = time.perf_counter()

    print(f"Time elapsed: {finish - start} second(s).")


