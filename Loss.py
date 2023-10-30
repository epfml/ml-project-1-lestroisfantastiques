import numpy as np
import matplotlib.pyplot as plt

def calculate_mse(e):
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    #compute loss by MSE
    e = y - tx.dot(w)
    loss =  calculate_mse(e)  
    return loss
