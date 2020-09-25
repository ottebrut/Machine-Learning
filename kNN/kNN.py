import pandas
import math
import numpy as np
import matplotlib.pyplot as plt


def dataset_min_max(dataset):
    min_max = []
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            break
        min_max.append([dataset[:, i].min(), dataset[:, i].max()])
    return min_max


def normalize(dataset, min_max):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:
                break
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])


def learning(dataset):
    windows = []
    f_scores = []
    max_f_score = answer = -1
    for k in range(1, len(dataset) - 1):
        cm = leave_one_out(dataset, k)
        f_scores.append(find_f_score(cm))
        windows.append(k)

        if f_scores[-1] > max_f_score:
            max_f_score = f_scores[-1]
            answer = k
    draw_plot(windows, f_scores)
    return answer


def leave_one_out(dataset, k):
    cm = []
    n = len(classes)
    for i in range(n):
        cm.append([0] * n)
    for i in range(len(dataset)):
        label = predict_with_window(np.concatenate((dataset[:i], dataset[i + 1:]), axis=0), dataset[i][:-1], k)
        cm[int(dataset[i][-1])][label] += 1
    return cm


def find_f_score(cm):
    n = len(classes)
    row_sum = [0] * n
    col_sum = [0] * n
    all = 0
    for i in range(n):
        for j in range(n):
            row_sum[i] += cm[i][j]
            col_sum[j] += cm[i][j]
            all += cm[i][j]
    f_score = 0
    for i in range(n):
        if col_sum[i] != 0:
            prec = cm[i][i] / col_sum[i]
        else:
            prec = 0
        if row_sum[i] != 0:
            recall = cm[i][i] / row_sum[i]
        else:
            recall = 0
        if (prec + recall) != 0:
            f_score += (2 * prec * recall / (prec + recall)) * row_sum[i]
    f_score /= all
    return f_score


# euclidean
def p(a, b):
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i])**2
    return math.sqrt(d)


# epanechnikov
def kernel(a):
    if abs(a) < 1:
        return 0.75 * (1 - a ** 2)
    return 0


eps = 1e-6


def predict_with_window(dataset, target, k):
    neighbours = []
    for row in dataset:
        neighbours.append([p(target, row[:-1]), row[-1]])
    neighbours.sort(key=lambda neighbour: neighbour[0])

    n = len(classes)
    window_param = neighbours[k][0]
    y_sum = [0] * n
    y_zero = [0] * n
    y_all = [0] * n
    sum = zeroes = False
    for neighbour in neighbours:
        y_all[int(neighbour[1])] += 1
        if window_param > eps:
            y_sum[int(neighbour[1])] += kernel(neighbour[0] / window_param)
            sum = True
        if neighbour[0] < eps:
            y_zero[int(neighbour[1])] = 1
            zeroes = True
    if sum:
        return y_sum.index(max(y_sum))
    elif zeroes:
        return y_zero.index(max(y_zero))
    else:
        return y_all.index(max(y_all))


def draw_plot(x, y):
    plt.plot(x, y)
    plt.show()


# https://www.openml.org/d/1500
# 3 classes; 8 features
filename = "seismic-bumps.csv"
dataset = pandas.read_csv(filename).values
target = [100, 50, 150, 75, 100, 50, 150]

min_max = dataset_min_max(dataset)
normalize(dataset, min_max)
normalized_target = target.copy()
normalize([normalized_target], min_max)

class_names = list(set(dataset[:, -1]))
classes = {}
for i in range(len(class_names)):
    classes[class_names[i]] = i
for row in dataset:
    row[-1] = classes[row[-1]]

k = learning(dataset)
label = predict_with_window(dataset, normalized_target, k)
print("For target: %s, Predicted: %s", target, class_names[label])
