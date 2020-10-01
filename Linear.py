import random
import numpy as np
import matplotlib.pyplot as plt


def read(n, x, y):
    for _ in range(n):
        x.append([int(i) for i in file.readline().split()])
        y.append(x[-1][-1])
        x[-1][-1] = 1


def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    return 0


def get_grad(n, x, y, w):
    grad = [0] * m
    eps = 1e-6
    for i in range(n):
        scalar = 0
        for j in range(m):
            scalar += w[j] * x[i][j]
        diff = scalar - y[i]
        upper_part = (sign(diff) * (abs(scalar) + abs(y[i])) - abs(diff) * sign(scalar))
        down_part = (abs(scalar) + abs(y[i]))**2
        if abs(down_part) > eps:
            c = upper_part / down_part
            for j in range(m):
                grad[j] += c * x[i][j]
    for j in range(m):
        grad[j] = grad[j] * 200 / n
    return grad


def gradient_descent(n, x, y, iterations):
    b = min(n, 10)
    c = 1e-2
    w0 = [0.0] * m
    w1 = [0] * m
    coord = []
    for i in range(n):
        coord.append(x[i])
        coord[-1].append(y[i])
    x = [0] * b
    y = [0] * b
    random.shuffle(coord)
    left = 0

    for t in range(1, iterations + 1):
        if left >= n:
            random.shuffle(coord)
            left = 0
        right = min(left + b, n)
        for i in range(left, right):
            x[i - left] = coord[i][:-1]
            y[i - left] = coord[i][-1]
        grad = get_grad(b, x, y, w0)
        mu = c / t
        for i in range(m):
            w1[i] = w0[i] - mu * grad[i]
            w0[i] = w1[i]
        left = right
    return w0


def genetic(n, x, y, iterations):
    c = 10
    e = 0.5
    w0 = [0.0] * m
    w1 = [0] * m
    smape_0 = smape(n, x, y, w0)

    for t in range(1, iterations + 1):
        mu = c / t
        for i in range(m):
            w1[i] = w0[i] - mu * random.uniform(-e, e)
        smape_1 = smape(n, x, y, w1)
        if smape_0 > smape_1:
            smape_0 = smape_1
            for i in range(m):
                w0[i] = w1[i]
    return w0


def smape(n, x, y, w):
    sum = 0
    for i in range(n):
        ft = 0
        for j in range(m):
            ft += w[j] * x[i][j]
        sum += 2 * abs(ft - y[i]) / (abs(ft) + abs(y[i]))
    return sum * 100 / n


file = open("./LR/3.txt", "r")
m = int(file.readline())
m += 1
dTrain_n = int(file.readline())
dTrain_x = []
dTrain_y = []
read(dTrain_n, dTrain_x, dTrain_y)
dTest_n = int(file.readline())
dTest_x = []
dTest_y = []
read(dTest_n, dTest_x, dTest_y)

iters = [10 ** i for i in range(2, 7)]

# МНК
v, d, ut = np.linalg.svd(dTrain_x, full_matrices=False)
C = 2.5
d = np.diag(d)
for i in range(len(d)):
    d[i][i] += C
tetta = np.matmul(np.matmul(np.matmul(np.transpose(ut), np.linalg.inv(d)), np.transpose(v)), dTrain_y)
plt.hlines(smape(dTest_n, dTest_x, dTest_y, tetta), iters[0], iters[-1])

# Градиентный спуск
gradient_smape = []
for iterations in iters:
    w = gradient_descent(dTrain_n, dTrain_x, dTrain_y, iterations)
    gradient_smape.append(smape(dTest_n, dTest_x, dTest_y, w))
plt.loglog(iters, gradient_smape, linestyle='-', marker='o', color='r')

# Генетический алгоритм
genetic_smape = []
for iterations in iters:
    w = genetic(dTrain_n, dTrain_x, dTrain_y, iterations)
    genetic_smape.append(smape(dTest_n, dTest_x, dTest_y, w))
plt.loglog(iters, genetic_smape, linestyle='-.', marker='x', color='g')
plt.show()
