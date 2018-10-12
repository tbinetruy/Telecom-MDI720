# Stats TP2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def import_data():
    df = pd.read_csv('invest.txt', delimiter=" ")
    return df

def log_scale(vect):
    return np.log(vect)

def normalize_df(data):
    data["gnp"] = log_scale(data["gnp"])
    data["invest"] = log_scale(data["invest"])

def plot_gnp_vs_invest(data):
    gnp = data["gnp"]
    invest = data["invest"]
    plt.figure()
    plt.plot(gnp, invest, color='blue', label='Father')
    plt.ylabel('Invest', fontsize=18)
    plt.xlabel('GNP', fontsize=18)
    plt.title("GNP vs Invest")
    plt.show()

def transverse(X):
    # x_{i,j} = x_{j,i}
    n = range(0, len(X))
    p = range(0, len(X[0]))
    X_t = [[0 for j in n] for i in p]
    for i in n:
        for j in p:
            X_t[j][i] = X[i][j]

    return X_t


def lin_reg(X, Y):
    # theta_hat = (X'X)^-1 * X'Y = A^-1 * B
    r = range(0, len(X))

    A = sum([X[i] * X[i] for i in r])
    B = sum([X[i] * Y[i] for i in r])
    theta_hat_1 = B / A
    x_bar = sum(X) / len(X)
    y_bar = sum(Y) / len(Y)
    theta_hat_0 = y_bar - theta_hat_1 * x_bar

    return [theta_hat_0, theta_hat_1]

def std_dev(vect):
    print(vect)
    vect_bar = sum(vect) / len(vect)
    diffs = [(v - vect_bar)**2 for v in vect]
    return math.sqrt(sum(diffs) / (len(vect)))

def mean(vect):
    return sum(vect) / len(vect)

def determination_coef(Y, Y_hat):
    Y_bar = mean(Y)
    r = range(0, len(Y))
    numerator = sum([(Y[i] - Y_hat[i])**2 for i in r])
    denominator = sum([(Y[i] - Y_bar)**2 for i in r])
    return 1 - numerator / denominator

def main():
    df = import_data()
    print(df)

    normalize_df(df)
    plot_gnp_vs_invest(df)

    gnp = df["gnp"]
    invest = df["invest"]

    theta_hat = lin_reg(gnp, invest)
    print("theta_hat: " + str(theta_hat))

    gnp_std = std_dev(gnp)
    invest_std = std_dev(invest)
    print("GNP std: " + str(gnp_std))
    print("numpy GNP std: " + str(np.std(gnp)))
    print("Invest std: " + str(invest_std))
    print("numpy invest std: " + str(np.std(invest)))

    invest_hat = [theta_hat[0] + x * theta_hat[1] for x in gnp]
    R2 = determination_coef(invest, invest_hat)
    print("R2: " + str(R2))


main()

# should equal [[1, 3, 5], [2, 4, 6]]
# print(transverse([[1,2],[3,4],[5,6]]))
