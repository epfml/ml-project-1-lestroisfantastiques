
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
#from plots import *
from Loss import *

#*****************-mean_squared_error_gd-*************************************


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
        loss = compute_mse(y, tx, w)  #loss = 1/(2*len(y)) * np.dot(e.T, e)

        #  update w by gradient
        w = w -gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
       

    return losses, ws[-1]



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
        loss = compute_mse(y, tx, w)
        ws.append(w)
        losses.append(loss)

    return losses, ws[-1]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stoch_grad=compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss=compute_mse(y, tx, w)    
        w=w-int(gamma)*stoch_grad
        losses.append(loss)
        ws.append(w)
        # ***************************************************
        

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
    mse = compute_mse(y, tx, w)
    return w, mse

#*****************-ridge_regression-*************************************

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y,tx,w)
    return w, mse

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


 def logistic_regression(y, tx, initial w, max iters, gamma ):
    # init parameters
    losses = []


    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss[-1], w



#*****************-reg_logistic_regression-*************************************


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):


    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w









# Compute the confusion matrix between two vectors, used to compare models.
def compute_confusion_matrix(true_values, predicted_values):
    '''Computes a confusion matrix using numpy for two np.arrays
    true_values and predicted_values. '''
    N = len(np.unique(true_values)) # Number of classes 
    result = np.zeros((N, N))
    true_values[np.where(true_values == -1)] = 0
    predicted_values[np.where(predicted_values == -1)] = 0

    for i in range(len(true_values)):
        result[int(true_values[i])][int(predicted_values[i])] += 1
        
    return result




def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]





def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):


    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """

    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
 
    # update w: 
    w_gd = w -gamma*grad
    # ***************************************************

    return loss, w_gd






def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    """
 
    N=len(y)
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + lambda_* 2 * w
    return loss, gradient

 



 def logistic_regression(y, tx, initial w, max iters, gamma ):
    # init parameters
    losses = []


    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss[-1], w




    def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """

    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w_gd = w -gamma*grad
    # ***************************************************
    #raise NotImplementedError
    return loss, w_gd