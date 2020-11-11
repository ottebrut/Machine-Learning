import pandas
import numpy as np
import sklearn.model_selection as m_s
import sklearn.metrics as metrics
import random
import math
import matplotlib.pyplot as plt

b = 0
C = kernel = alphas = []
eps = 1e-6


def get_diff(index, y):
    n = y.shape[0]
    point = 0
    for j in range(n):
        point += alphas[j] * kernel[index][j] * y[j]
    return point - y[index]


def take_step(i1, i2, y, c):
    if y[i1] != y[i2]:
        l = max(0, alphas[i2] - alphas[i1])
        h = min(c, alphas[i2] - alphas[i1] + c)
    else:
        l = max(0.0, alphas[i2] + alphas[i1] - c)
        h = min(c, alphas[i2] + alphas[i1])
    if abs(l - h) < eps:
        return

    eta = (kernel[i1][i1] + kernel[i2][i2]) - 2 * kernel[i1][i2]
    e1 = get_diff(i1, y)
    e2 = get_diff(i2, y)
    s = y[i1] * y[i2]
    if eta > 0:
        a2 = alphas[i2] + (e1 - e2) * y[i2] / eta
        a2 = max(a2, l)
        a2 = min(a2, h)
    else:
        return
    if abs(a2 - alphas[i2]) < eps:
        return
    alphas[i1] += s * (alphas[i2] - a2)
    alphas[i2] = a2


def get_alphas(y, c):
    n = y.shape[0]
    global alphas
    indexes = [0] * n
    alphas = [0.0] * n
    for i in range(n):
        indexes[i] = i
    iterations = 2000

    for iter in range(iterations):
        np.random.shuffle(indexes)
        for i in indexes:
            j = math.floor(random.randrange(0, n))
            while j == i:
                j = math.floor(random.randrange(0, n))
            take_step(i, j, y, c)

    global b
    avg_b = cnt = 0
    for i in range(n):
        if alphas[i] > eps:
            avg_b += -get_diff(i, y)
            cnt += 1
    if cnt != 0:
        b = avg_b / cnt
    else:
        b = 0
    return


def count_kernel(x1, x2, kernel_type, degree, beta):
    n = x1.shape[0]
    m = x2.shape[0]
    for i in range(n):
        for j in range(m):
            if kernel_type == 0:
                # linear function
                kernel[i][j] = np.dot(x1[i], x2[j])
            elif kernel_type == 1:
                # polynomial
                kernel[i][j] = np.power(np.dot(x1[i], x2[j]), degree)
            else:
                # gaussian
                kernel[i][j] = np.exp(-beta * np.power(np.linalg.norm(x1[i] - x2[j]), 2))


def predict(x_test, x_train, y_train, kernel_type, degree, beta):
    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    count_kernel(x_test, x_train, kernel_type, degree, beta)
    y_predicted = np.arange(n_test)
    for i in range(n_test):
        sm = b
        for j in range(n_train):
            sm += alphas[j] * kernel[i][j] * y_train[j]
        y_predicted[i] = 1 if sm > 0 else -1
    return y_predicted


def get_score(x, y, c, kernel_type, degree=0, beta=0):
    k = 3
    k_fold = m_s.KFold(n_splits=k)
    accuracy_scores = cnt = 0
    for train_index, test_index in k_fold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count_kernel(x_train, x_train, kernel_type, degree, beta)
        get_alphas(y_train, c)
        y_predicted = predict(x_test, x_train, y_train, kernel_type, degree, beta)
        accuracy_scores += metrics.accuracy_score(y_test, y_predicted)
        print(accuracy_scores)
        cnt += 1
    print("------------------")
    print(accuracy_scores / cnt)
    print("------------------")
    return accuracy_scores / cnt


def update_parameters(parameters, new_score, c, kernel_type, degree=0, beta=0):
    if new_score > parameters["score"]:
        parameters["score"] = new_score
        parameters["c"] = c
        parameters["kernel_type"] = kernel_type
        parameters["degree"] = degree
        parameters["beta"] = beta


def find_parameters(x, y):
    global C, kernel
    C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    kernel = []
    n = x.shape[0]
    for i in range(n):
        kernel.append([0] * n)
    kernel_types = [0, 1, 2]
    degrees = [2, 3, 4, 5]
    betas = [1, 2, 3, 4, 5]

    parameters = {"score": -1}
    for c in C:
        for kernel_type in kernel_types:
            if kernel_type == 0:
                update_parameters(parameters, get_score(x, y, c, kernel_type), c, kernel_type)
            elif kernel_type == 1:
                for degree in degrees:
                    update_parameters(parameters, get_score(x, y, c, kernel_type, degree), c, kernel_type, degree)
            else:
                for beta in betas:
                    update_parameters(parameters, get_score(x, y, c, kernel_type, beta=beta), c, kernel_type, beta=beta)
    return parameters


def read_data(file):
    data = pandas.read_csv(file).values
    x = data[:, :-1]
    y = data[:, -1]
    for (i, cl) in enumerate(y):
        y[i] = 1 if cl == 'P' else -1
    return x, np.array(y, dtype=int)


def shuffle(x, y):
    n = x.shape[0]
    indexes = [0] * n
    for i in range(n):
        indexes[i] = i
    random.shuffle(indexes)
    return x[indexes], y[indexes]


def get_class(point, x, y, parameters):
    sm = b
    n = y.shape[0]
    count_kernel(np.array([point]), x, parameters["kernel_type"], parameters["degree"], parameters["beta"])
    for j in range(n):
        sm += alphas[j] * kernel[0][j] * y[j]
    return 1 if sm > 0 else -1


def show(x, y, parameters, file):
    X, Y = np.mgrid[0:25:complex(0, 100), 0:6:complex(0, 100)]
    matrix = []
    for i in range(len(X)):
        matrix.append([])
        for j in range(len(Y)):
            matrix[i].append(get_class([float(X[i][0]), float(Y[0][j])], x, y, parameters))
    fig, ax0 = plt.subplots()
    c = ax0.pcolor(X, Y, matrix, cmap='rainbow', vmin=min(min(matrix)), vmax=max(max(matrix)))
    fig.colorbar(c, ax=ax0)
    n = x.shape[0]
    for i in range(n):
        if get_class(x[i], x, y, parameters) * y[i] > 0:
            plt.plot(x[i][0], x[i][1], '+')
        else:
            plt.plot(x[i][0], x[i][1], '_')
    plt.savefig(file + '.png')
    plt.show()


def solve_svm(file):
    x, y = read_data(file)
    x, y = shuffle(x, y)
    parameters = find_parameters(x, y)
    print(parameters)
    count_kernel(x, x, parameters["kernel_type"], parameters["degree"], parameters["beta"])
    get_alphas(y, parameters["c"])
    show(x, y, parameters, file)


# {'score': 0.8386752136752137, 'c': 10.0, 'kernel_type': 2, 'degree': 0, 'beta': 1}
solve_svm("chips.csv")
# {'score': 0.9009009009009009, 'c': 0.5, 'kernel_type': 0, 'degree': 0, 'beta': 0}
solve_svm("geyser.csv")
