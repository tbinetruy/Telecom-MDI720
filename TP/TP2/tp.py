# Stats TP2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_data():
    df = pd.read_csv('invest.txt', delimiter=" ")
    return df


def plot_gnp_vs_invest(data):
    plt.figure()
    plt.plot(data["gnp"], data["invest"], color='blue', label='Father')
    plt.ylabel('Invest', fontsize=18)
    plt.xlabel('GNP', fontsize=18)
    plt.title("GNP vs Invest")
    plt.show()


df = import_data()
print(df)
plot_gnp_vs_invest(df)
