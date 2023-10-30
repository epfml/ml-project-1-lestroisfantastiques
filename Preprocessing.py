import numpy as np
from helpers import *

def select_features(data,indices):
    features = []

   for idx in indices:

        # Select the specific feature column in the randomly selected data values
        feature_column = data[:, idx-2]  # Utilisez feature_idx-2 car les indices commencent à 0 et il y a un decalage de 1 avec les ids
        # Add the column to the list features
        features.append(feature_column)
       
        # List into Numpy Array
    #data_selected= np.array(features)
    data_selected = np.column_stack(features)

     #taking the corresponding headers
    #Headers = [headers[idx-1] for idx in indices]
    return data_selected



def balance_data(X,Y, X_test):
    # Identify indices of patients with (y=1) or without heart attack(y=0)
    indices_y1 = np.where(Y == 1)[0]
    indices_y0 = np.where(Y == 0)[0]

    # Selecting the same number of random indices for each class 
    num_samples = min(len(indices_y1), len(indices_y0))
    selected_indices_y1 = np.random.choice(indices_y1, num_samples, replace=False)
    selected_indices_y0 = np.random.choice(indices_y0, num_samples, replace=False)

    # Combining the indices for a balanced data 
    selected_indices = np.concatenate((selected_indices_y1, selected_indices_y0))

    #Selecting the same number of data in x_test than in x_train 
    idx_rdm = np.random.choice(X_test.shape[0], size=len(selected_indices), replace=False)

    # Selecting the corresponding data 
    balanced_X = X[selected_indices]
    balanced_Y = Y[selected_indices]
    balanced_X_test = X_test[idx_rdm, :]
 

    return balanced_X, balanced_Y,balanced_X_test


def standardize_clean_dataframe(X):
    '''
    Input = 2 dimensional array of features
    Output = updated 2 dimensional array of features
    This function does several things 
    1) It replaces the -999 values by the median value of the column (computed without taking the -999 values into account)
    2) It standardises the data, i.e for each column we substract by the mean and divide by the standard deviation '''  
    dataframe = X.copy()
    n = dataframe.shape[1]
    for i in range(n):
        column = dataframe[:, i].copy()      
        column = np.where(column == 9| np.isnan(column), np.mean(column[column != 9]), column)
        dataframe[:, i] = column.copy()
        
    Mean = np.mean(dataframe, axis = 0)
    stand_dev = np.std(dataframe, axis = 0)
    
    matrix_of_mean = np.full(dataframe.shape, Mean)
    matrix_of_std = np.full(dataframe.shape, stand_dev)
    return (dataframe - matrix_of_mean)/(matrix_of_std)



def clean_data(X):
    '''
    Input = 2 dimensional array of features
    Output = updated 2 dimensional array of features
    This function replaces the Nan or 9 values by the mean value of the column (computed without taking the -999 values into account)
 '''  
    dataframe = X.copy()
    n = dataframe.shape[1]
    for i in range(n):
        column = dataframe[:, i].copy()      
        column = np.where(column == 9| np.isnan(column), np.mean(column[column != 9]), column)
        dataframe[:, i] = column.copy()

    return dataframe     
  


def standardize(x):
    """Standardize the original data set:for each column we substract by the mean and divide by the standard deviation""" 
    Mean = np.mean(x, axis = 0)
    stand_dev = np.std(x, axis = 0)
    
    matrix_of_mean = np.full(x.shape, Mean)
    matrix_of_std = np.full(x.shape, stand_dev)
    
    return (x - matrix_of_mean)/(matrix_of_std)


def split_data(x, y, ratio = 0.8, seed = 1):
    """split the dataset based on the split ratio """
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row)) 
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te





