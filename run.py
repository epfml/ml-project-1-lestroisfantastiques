# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2


from helpers import load_csv_data

# load data.
PATH = "data/dataset/"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset/", sub_sample=False)



#height_weight_ratio, filling nan with mean
weight_ratio = x_train[:,252]
weight_ratio_filled = weight_ratio
weight_ratio_filled[np.isnan(weight_ratio_filled)]=np.mean(weight_ratio_filled[~np.isnan(weight_ratio_filled)])


#age5y, filling nan with mean
age = x_train[:,245]
age_filled = age
age_filled[np.isnan(age_filled)]=np.mean(age_filled[~np.isnan(age_filled)])


x_train = np.c_[weight_ratio_filled, age_filled]


#Selecting an equal number of people with heart attack and those who has not

# Identifiez les indices des patients avec un arrêt cardiaque (y=1) et sans (y=0)
indices_y1 = np.where(y_train == 1)[0]
indices_y0 = np.where(y_train == 0)[0]






num_samples = min(len(indices_y1), len(indices_y0))
selected_indices_y1 = np.random.choice(indices_y1, num_samples, replace=False)
selected_indices_y0 = np.random.choice(indices_y0, num_samples, replace=False)

print('indices selectionés:', selected_indices_y1)
print('indices selectionés y0:', selected_indices_y0)


selected_indices = np.concatenate((selected_indices_y1, selected_indices_y0))


balanced_x_train = x_train[selected_indices]
balanced_y_train = y_train[selected_indices]



from helpers import standardize
balanced_x_train_std = np.c_[standardize(balanced_x_train[:,0]).reshape(-1), standardize(balanced_x_train[:,1]).reshape(-1)]



X = balanced_x_train_std
y = balanced_y_train
# Specify the proportion of data to use for the test set (in this example, 20%)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))

# Split the data into training and testing sets
X_train, X_test_in = X[:split_index], X[split_index:]
y_train, y_test_in = y[:split_index], y[split_index:]

# You can now use X_train and y_train to train your model
# and X_test and y_test to evaluate the model's performance.



from implementations import *
from helpers import build_poly
from cross_validation import *
from Loss import test_pred



#test 

X_poly = build_poly(X_train, 3)

#mean_squarred_error_gd
initial_w = np.zeros(7)
loss_MSE_gd, w_opt_MSE_gd = mean_squarred_error_gd(y_train, X_poly, initial_w, 50, 0.003)
#loss diverge



##mean_squarred_error_sgd
initial_w = np.zeros(7)
loss_MSE_sgd, w_opt_MSE_sgd = mean_squared_error_sgd(y_train, X_poly, initial_w ,50, 0.003)

#lossdiverge


#Least_squares
w_opt_LR, loss_LR = least_squares(y_train, X_poly)



#Ridge Regression
best_lambda, best_rmse = cross_validation_demo(3, 3, np.logspace(-4, 0, 30), y_train, X_train)
w_opt_RR, mse_RR = ridge_regression(y_train, X_poly, best_lambda)


#test
X_test_poly = build_poly(X_test_in, 3)

y_pred_GD = np.dot(X_test_poly, np.array(w_opt_MSE_gd))
y_pred_SGD = np.dot(X_test_poly, np.array(w_opt_MSE_sgd))
y_pred_LR = np.dot(X_test_poly, w_opt_LR)
y_pred_RR = np.dot(X_test_poly, w_opt_RR)


accuracy_GD = test_pred(y_pred_GD, y_test_in)

XTEST1 = np.c_[x_test[:, 245], x_test[:, 252]]
XTEST2 = build_poly(XTEST1, 3)


y_pred_final = np.dot(XTEST2, w_opt_MSE_GD)

y_pred_final[y_pred_final<0.5]=-1
y_pred_final[y_pred_final>0.5]=1
y_pred_final[np.isnan(y_pred_final)]=-1
create_csv_submission(test_ids, y_pred_final, "final prediction")





