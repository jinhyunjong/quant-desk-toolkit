import numpy as np

def discount_factor(rate, t):
    return np.exp(-rate * t)

def zero_rate(df, t):
    return -np.log(df) / t
