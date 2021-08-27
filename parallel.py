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
    clf = MLPClassifier(30, activation="tanh", learning_rate_init=0.002, max_iter=200, random_state=0)
    clf.fit(data[0], data[1])
    return clf


if __name__ == '__main__':
    raw_data = pd.read_csv("D:\\pidp dataset\\airline_passenger_satisfaction.csv")
    raw_data = raw_data.dropna()

    array = np.asarray(raw_data)

    encoder = OrdinalEncoder()
    data = encoder.fit_transform(array)
    data = pd.DataFrame(data)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    x = np.asarray(x)
    y = np.asarray(y)

    x_test = x[int(len(x)*0.9):]
    y_test = y[int(len(y)*0.9):]

    x1 = x[:int(len(x) * 0.25)]
    x2 = x[int(len(x) * 0.25):int(len(x) * 0.5)]
    x3 = x[int(len(x) * 0.5):int(len(x) * 0.75)]
    x4 = x[int(len(x) * 0.75):]

    y1 = y[:int(len(y) * 0.25)]
    y2 = y[int(len(y) * 0.25):int(len(y) * 0.5)]
    y3 = y[int(len(y) * 0.5):int(len(y) * 0.75)]
    y4 = y[int(len(y) * 0.75):]


    data = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    clfs = list()

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        procs = [executor.submit(train_neural_net, data[i]) for i in range(len(data))]
        for i in range(len(procs)):
            clfs.append(procs[i].result())


    acc = 0

    for x, y in zip(x_test, y_test):
        tmp = []

        tmp.append(int(clfs[0].predict([x])[0]))
        tmp.append(int(clfs[1].predict([x])[0]))
        tmp.append(int(clfs[2].predict([x])[0]))
        tmp.append(int(clfs[3].predict([x])[0]))



        count = Counter(tmp)
        pred = count.most_common()

        if pred[0][0] == y:
            acc += 1

    print(f"Accuracy of the parallel neural nets is {acc/len(y_test)}")

    finish = time.perf_counter()

    print(f"Time elapsed: {finish - start} second(s).")
    

