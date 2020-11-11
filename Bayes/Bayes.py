from os import listdir
import sklearn.metrics as metrics
import math
import matplotlib.pyplot as plt

INF = 2e9
MAX = 700


def get_words_probs(w, count_y, alpha):
    count_x = [{}, {}]
    calculated = set()
    for k in range(2):
        for word_set in w[k]:
            for word in word_set:
                if not (word in count_x[k]):
                    count_x[k][word] = 0
                count_x[k][word] += 1
                calculated.add(word)

    words = [{}, {}]
    for word in calculated:
        for k in range(2):
            if not (word in count_x[k]):
                count_x[k][word] = 0
            words[k][word] = (alpha + count_x[k][word]) / (2. * alpha + count_y[k])
    return words


def get_p(words, count_y, lambdas, word_set):
    n = count_y[0] + count_y[1]
    p = [0, 0]
    mn = INF
    mx = -INF
    for k in range(2):
        p[k] = math.log(count_y[k] / n)
        for word in words[k]:
            if word in word_set:
                p[k] += math.log(words[k][word])
            else:
                p[k] += math.log(1. - words[k][word])
        p[k] += math.log(lambdas[k])
        mn = min(mn, p[k])
        mx = max(mx, p[k])

    p_word = 0
    for k in range(2):
        p[k] -= (mn + mx) / 2
        if p[k] < MAX:
            p_word += math.exp(p[k])
    return p, p_word


def get_class(words, count_y, lambdas, word_set):
    p, p_word = get_p(words, count_y, lambdas, word_set)

    mx = 0
    cl = 0
    for k in range(2):
        if p[k] < MAX and p_word != 0:
            prob = math.exp(p[k]) / p_word
        else:
            prob = INF
        if prob > mx:
            mx = prob
            cl = k
    return cl


def get_accuracy(w, count_y, alpha, lambda_legit):
    accuracy = 0
    cnt = 0
    # k_fold
    for i in range(10):
        w_i = [[], []]
        count_y_i = [0, 0]
        for j in range(10):
            if j != i:
                for k in range(2):
                    for s in w[i][k]:
                        w_i[k].append(s)
                    count_y_i[k] += count_y[i][k]

        words = get_words_probs(w_i, count_y_i, alpha)

        y_test = []
        y_pred = []
        for k in range(2):
            for word_set in w[i][k]:
                y_test.append(k)
                y_pred.append(get_class(words, count_y_i, [1, lambda_legit], word_set))
        accuracy += metrics.accuracy_score(y_test, y_pred)
        cnt += 1
    return accuracy / cnt


def n_gramms(words, n=1):
    new_words = []
    for i in range(len(words) - n):
        word = words[i]
        for j in range(i + 1, i + n):
            word += words[i]
        new_words.append(word)
    return new_words


def read_data(directory):
    w = [[], []]
    count_y = [0, 0]
    files = [directory + file_name for file_name in listdir(directory)]
    for file_name in files:
        cl = 1 if "legit" in file_name else 0
        file = open(file_name, "r")
        header = file.readline().split()
        header = header[1:]
        for i in range(len(header)):
            header[i] += "$"
        file.readline()
        words = header + file.readline().split()

        w[cl].append(set(n_gramms(words, 1)))
        count_y[cl] += 1
    return w, count_y


def solve_bayes():
    w = []
    count_y = []
    for i in range(1, 11):
        w_i, count_y_i = read_data("part" + str(i) + "/")
        w.append(w_i)
        count_y.append(count_y_i)

    alpha = 1
    max_accuracy = -1
    best_lambda = 0
    for lambda_legit in range(1, 5):
        accuracy = get_accuracy(w, count_y, alpha, 10**lambda_legit)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_lambda = 10**lambda_legit
        print(accuracy, 10**lambda_legit)
        f = True
        for i in range(10):
            for word_set in w[i][1]:
                cl = get_class(get_words_probs(w[i], count_y[i], alpha), count_y[i], [1, 10**lambda_legit], word_set)
                if cl != 1:
                    f = False
        if f:
            print(10**lambda_legit)
    print(max_accuracy, best_lambda)

    ps = [[], []]
    for k in range(2):
        for i in range(10):
            ps[k].append([])
            for word_set in w[i][k]:
                p, _ = get_p(get_words_probs(w[i], count_y[i], alpha), count_y[i], [1, best_lambda], word_set)
                ps[k][i].append(p[0])

    spam_p = []
    for k in range(2):
        for i in range(10):
            for j in range(len(w[i][k])):
                spam_p.append(ps[k][i][j])
    spam_p = list(set(spam_p))
    spam_p.sort()
    spam_p.reverse()
    spec = []
    sens = []
    for spam in spam_p:
        tp = pos = fp = neg = 0
        for k in range(2):
            for i in range(10):
                for j in range(len(w[i][k])):
                    p = ps[k][i][j]
                    tp += (p > spam and k == 0)
                    fp += (p > spam and k == 1)
                    pos += (k == 0)
                    neg += (k == 1)
        spec.append(fp / neg)
        sens.append(tp / pos)
    plt.plot(spec, sens)
    plt.show()


solve_bayes()
