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

def main():
    df = import_data()
    normalize_df(df)
    plot_gnp_vs_invest(df)

main()
