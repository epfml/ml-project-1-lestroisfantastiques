def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
    
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum
    """
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def test_pred(y_pred, y_test):
    y_pred[y_pred<0.5]=0
    y_pred[y_pred>=0.5]=1
    return sum(y_pred==y_test)/len(y_test)