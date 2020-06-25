# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:29:56 2018
@authors: Sabrina & Marcio
"""
from sklearn.svm import NuSVR
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
  
#Beta = [1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20]
def prep_process(modelo, X_train, y_train, X_test_n, y_test_n):
    val = modelo.fit(X_train, y_train)        # Fit the model according to the given training data.
    y_regr_pred = modelo.predict(X_test_n)    # Perform regression samples in X_test.    
    R2_score = val.score(X_test_n, y_test_n)  # Returns the mean accuracy on the given test data and labels
    rmse = mean_squared_error(y_test_n, y_regr_pred)  # The best value is 0             
    return(R2_score, rmse)
          
def regress_knn(n_neighbors, X_train, y_train, X_test_n, y_test_n):
    knn = KNeighborsRegressor(n_neighbors, weights='distance')      # Simple method for creating regressions that aren’t linear    
    regr_knn = prep_process(knn, X_train, y_train, X_test_n, y_test_n)
    return(regr_knn[0], regr_knn[1]) 
   
def regress_NuSVR(X_train, y_train, X_test_n, y_test_n, C1, nu1): 
    nusvr = NuSVR(nu = nu1, C = C1, kernel= 'rbf', gamma=0.0001, tol = 0.001)
    regr_nusvr = prep_process(nusvr, X_train, y_train, X_test_n, y_test_n)
    return(regr_nusvr[0], regr_nusvr[1])

def parameter_choosing_svr(estimator, params, X_train, Y_train, tem_list):
    grid_search = GridSearchCV(estimator, param_grid=params, cv=3)
    grid_search.fit(X_train, Y_train)
    for k,v in grid_search.best_params_.items():
        tem_list.append(v)
        print('', v)    
    return(tem_list[0], tem_list[1])

estimator = NuSVR(kernel='rbf', gamma=0.0001, tol=0.001) 
Cs = np.arange(10, 30, 5)
Nus = np.arange(0.2, 0.9, 0.1)
params = {'nu' : Nus, 'C': Cs}

# Read data files
fname_pars = '/Predator_prey_equation/File2_CSV/train_Beta_Parameter_values.csv' 
par_values = pd.read_csv(fname_pars)
fname_pars1 = '/Predator_prey_equation/File2_CSV/test_Beta_Parameter_values.csv' 
par_values1 = pd.read_csv(fname_pars1) 

for j in range(3,20):
    fname_features = '/Predator_prey_equation/File2_CSV/train_TDA_Beta_features_m_%d.csv' %(j)  
    fname_features1 = '/Predator_prey_equation/File2_CSV/test_TDA_Beta_features_m_%d.csv' %(j)  
    print('file number', j)
    
    # Read data files
    tda_features = pd.read_csv(fname_features)   
    tda_features1 = pd.read_csv(fname_features1)
 
    # Get X and y
    X = np.array(tda_features.loc[:, 'B1_1':"B2_%d"%(j)])
    y = np.array(par_values.loc[:, 'Beta']) 

    XX = np.array(tda_features1.loc[:, 'B1_1':"B2_%d"%(j)])
    yy = np.array(par_values1.loc[:, 'Beta'])

    matrix1 = np.zeros((30,2))
    matrix2 = np.zeros((30,4))

    for time in range(0,30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_test_n = np.concatenate((X_test,XX))
        y_test_n = np.concatenate((y_test,yy)) 
    
        modelknn = regress_knn(10, X_train, y_train, X_test_n, y_test_n)
        matrix1[time,0] = modelknn[0] # R^2
        matrix1[time,1] = modelknn[1] # rmse
    
        tem_list =[]
        parm1 = parameter_choosing_svr(estimator, params, X_train, y_train, tem_list)    
        if parm1[0]<1:
            par_nu = parm1[0]
            par_C = parm1[1]
        else:
            par_nu = parm1[1]
            par_C = parm1[0]    
        
        modelsvr = regress_NuSVR(X_train, y_train, X_test_n, y_test_n, par_C, par_nu)
        matrix2[time,0] = modelsvr[0] # R^2
        matrix2[time,1] = modelsvr[1] # rmse
        matrix2[time,2] = par_C       # C parameter
        matrix2[time,3] = par_nu      # Nu paramter
          
        print('time', time)
        
    final = np.array(matrix1)
    np.savetxt("/Predator_prey_equation/Results-files2/R2_rmse/knn_measures_m%d.txt"%(j), final, fmt="%.14f", delimiter=",")

    final1 = np.array(matrix2)
    np.savetxt("/Predator_prey_equation/Results-files2/R2_rmse/svr_measures_m%d.txt"%(j), final1, fmt="%.14f", delimiter=",")
print('end')
