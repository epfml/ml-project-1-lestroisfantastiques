import numpy as np

from implementations import *
from helpers import *

from Preprocessing import *

# Writing the paths of our csv files
PATH = "data/dataset/"


# Loading data
print('Loading data...')
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("PATH", sub_sample=False)

#load headers
headers = load_headers(PATH)


# Preprocessing data
print('Prepocessing data...')

#selecting features
indices= [231,232,233,235,238,239,237,247,255,260,264,265,266,277,278,288,302,314,320]
tx = select_features(x_train,headers,indices)

#balance data
tx, y, x_te = balance_data(tx,y_train, x_test)

#clean data of missing values
tx = clean_data(tx)
tx_test = clean_data(x_te)

#standardize data
#**************


# Augmenting features :
print('Expanding features...')
#**************
#tx = expand_features_angles(tx)
#tx_test = expand_features_angles(tx_test)

# Splitting the data between training and validation
print('Splitting dataset between training and validation subsets...')
x_tr, x_val, y_tr, y_val = split_data(tx, y, 0.9)








# *****************************************************
