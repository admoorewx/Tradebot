import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(X, Y, xlabel, ylabel, title=""):
    plt.figure()
    plt.scatter(X, Y, marker='o', color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()


def plot_line(X, Y, xlabel, ylabel, title=""):
    plt.figure()
    plt.plot(X, Y, color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()


def plot_lines(Y1, Y2, xlabel, ylabel, title=""):
    plt.figure()
    plt.plot(np.arange(0, len(Y1)), Y1, color='k')
    plt.plot(np.arange(0, len(Y2)), Y2, color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()


def plot_histogram(Y):
    plt.figure()
    # plt.hist(Y,bins=np.arange(-10.0,10.0,0.5))
    plt.hist(Y)
    plt.show()