from matplotlib import pyplot as plt
import numpy as np
import pygame as pg

from scipy.interpolate import interp1d

model = [0, 0.075, 1/4, 1/2, 3/4, 7/8, 15/16, 31/32, 63/64, 127/128, 255/256, 1]

f = lambda x: x

def use(model):
    global f
    f = interp1d(np.linspace(0, 1, len(model)), model, "quadratic")

def show(x, y):
    plt.scatter(x, y, s=10)
    plt.plot(x, y, c="red")
    return plt.show()

def show_f(n=32):
    return show_map(0, 1, n)

def show_map(start, end, n):
    x = np.linspace(0, 1, n)
    y = [get_value(start, end, ratio) for ratio in x]
    return show(x, y)

def get_value(start, end, ratio):
    if ratio == 1:
        return end
    else:
        return start + (end-start) * f(ratio)
