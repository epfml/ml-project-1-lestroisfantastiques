import numpy as np
from helpers import build_poly
from implementations import ridge_regression
from Loss import compute_mse

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

    
   
    
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,P) where P are the different columns we use
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """


    
    for i in range(x.shape[1]):
        xi=x[:,i]
        x_tr, y_tr, x_te, y_te = ([] for i in range(4))
        for kf in range(k_indices.shape[0]):
            
            if (kf == k):
                x_te.append((xi[k_indices[kf]]).tolist())
                y_te.append((y[k_indices[kf]]).tolist())
            else:
                x_tr.append((xi[k_indices[kf]]).tolist())
                y_tr.append((y[k_indices[kf]]).tolist())
            
            
        #give them the appropriate shape for treatment
        x_tr, y_tr = np.array(x_tr).flatten(), np.array(y_tr).flatten()
        x_te, y_te = np.array(x_te).flatten(), np.array(y_te).flatten()
        if i==0:
            x_tr_f = x_tr
            x_te_f = x_te
        else :    
            x_tr_f = np.c_[x_tr_f, x_tr]
            x_te_f = np.c_[x_te_f, x_te]


    # form data with polynomial degree: 
    tx_tr = build_poly(x_tr_f,degree)
    tx_te = build_poly(x_te_f,degree)

    # ridge regression: 
    w, mse = ridge_regression(y_tr, tx_tr, lambda_)
    
    # calculate the loss for train and test data: 
    loss_tr, loss_te = np.sqrt(2*compute_mse(y_tr,tx_tr,w)), np.sqrt(2*compute_mse(y_te,tx_te,w))
    # ***************************************************
    return loss_tr, loss_te



def cross_validation_demo(degree, k_fold, lambdas, y, x):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

   
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over lambdas: TODO
    rtr = []
    rte = []
    for lambda_ in lambdas:
        for kf in range(k_fold):
            rtr.append(cross_validation(y,x,k_indices,kf,lambda_,degree)[0])
            rte.append(cross_validation(y,x,k_indices,kf,lambda_,degree)[1])
        rmse_tr.append(sum(rtr)/len(rtr))
        rmse_te.append(sum(rte)/len(rte))
        rtr.clear()
        rte.clear()
    
    best_rmse = min(rmse_te)
    best_lambda = lambdas[rmse_te.index(min(rmse_te))]
    # ***************************************************

    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse