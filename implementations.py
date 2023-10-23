
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from plots import *

#*****************-mean_squared_error_gd-*************************************

def calculate_mse(e):
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    #compute loss by MSE
    e = y - tx.dot(w)
    loss =  calculate_mse(e)  
    return loss
        
   

def compute_gradient(y, tx, w):
    #compute gradient vector    
    N = len(y)    
    e = y - tx.dot(w) 
    gradient = -1/N * tx.T.dot(e)
    
    return gradient

   



def mean_squarred_error_gd(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    gradient = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss        
        gradient = compute_gradient(y,tx,w)      
        loss = compute_loss(y, tx, w)  #loss = 1/(2*len(y)) * np.dot(e.T, e)

        #  update w by gradient
        w = w -gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
       
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws

#*****************-mean_squared_error_sgd-*************************************


def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e    


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        grad, _ = compute_stoch_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w,True)
        ws.append(w)
        losses.append(loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws

#*****************-least_squares-*************************************

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse

#*****************-ridge_regression-*************************************

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

#*****************-logistic_regression-*************************************


def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]


    a = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(a)) + (1 - y).T.dot(np.log(1 - a))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


def calculate_gradient(y, tx, w):
    a = sigmoid(tx.dot(w))
    grad = tx.T.dot(a - y) * (1 / y.shape[0])
    return grad

def calculate_hessian(y, tx, w):
    a = sigmoid(tx.dot(w))
    a = np.diag(a.T[0])
    r = np.multiply(a, (1 - a))
    return tx.T.dot(r).dot(tx) * (1 / y.shape[0])


def learning_by_newton_method(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    for n_iter in range(max_iters) :
        w = w - gamma * np.linalg.solve(hessian, gradient)
    
    return loss, w

#*****************-reg_logistic_regression-*************************************

def learning_by_penalized_gradient(y, tx,lambda_, initial_w, max_iters, gamma):
    w = initial_w
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + lambda_ * w**2
    for n_iter in range(max_iters):
        w = w - gamma * gradient
   
    return loss, w

