# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:20:43 2018
@authors: Sabrina & Marcio
"""
from sklearn.svm import NuSVR
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

#Beta = [1.0, 1.2, 1.4, 1.6, 1.8]
def prep_process(modelo, X_train, X_test, y_train, y_test):
    val = modelo.fit(X_train, y_train)     # Fit the model according to the given training data.
    y_regr_pred = modelo.predict(X_test)   # Perform regression samples in X_test.
    R2_score = val.score(X_test, y_test)   # Returns the R2 score on the given test data and labels
    rmse = mean_squared_error(y_test, y_regr_pred)  # The best value is 0   
    return(R2_score, rmse)
    
def regress_knn(n_neighbors, X_train, X_test, y_train, y_test):  
    knn = KNeighborsRegressor(n_neighbors, weights='distance')   # Simple method for creating regressions that aren’t linear    
    regr_knn = prep_process(knn, X_train, X_test, y_train, y_test)
    return(regr_knn[0], regr_knn[1]) 
    
def regress_NuSVR(X_train, X_test, y_train, y_test, C1, nu1): 
    nusvr = NuSVR(nu = nu1, C = C1, kernel= 'rbf', gamma=0.0001, tol = 0.001)
    regr_nusvr = prep_process(nusvr, X_train, X_test, y_train, y_test)
    return(regr_nusvr[0], regr_nusvr[1])

def parameter_choosing_svr(estimator, params, X_train, Y_train, tem_list):
    grid_search = GridSearchCV(estimator, param_grid=params, cv=5)
    grid_search.fit(X_train, Y_train)
    for k,v in grid_search.best_params_.items():
        tem_list.append(v)
        print('', v)    
    return(tem_list[0], tem_list[1])
      
estimator = NuSVR(kernel='rbf', gamma=0.0001, tol=0.001) 
Cs = np.arange(5, 30, 5)
Nus = np.arange(0.2, 0.9, 0.1)
params = {'nu' : Nus, 'C': Cs}

# Read data files
fname_pars = '/Ginzburg_Landau_equation/File4_CSV/Beta_Parameter_values.csv'
par_values = pd.read_csv(fname_pars)

for j in range(10,50):
    fname_features = '/Ginzburg_Landau_equation/File4_CSV/TDA_Beta_features_m_%d.csv' %(j)
    print('file number', j)

    # Read data files
    tda_features = pd.read_csv(fname_features)
    
    matrix1 = np.zeros((30,2))
    matrix2 = np.zeros((30,4))

    # Get X and y
    X = np.array(tda_features.loc[:, 'B1_1':"B2_%d"%(j)])
    y = np.array(par_values.loc[:, 'Beta'])

    for time in range(0,30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        modelknn = regress_knn(5, X_train, X_test, y_train, y_test)
        matrix1[time,0] = modelknn[0]  # R^2 result
        matrix1[time,1] = modelknn[1]  # rmse result
        
        tem_list =[]
        parm1 = parameter_choosing_svr(estimator, params, X_train, y_train, tem_list)
        if parm1[0]<1:
            par_nu = parm1[0]
            par_C = parm1[1]
        else:
            par_nu = parm1[1]
            par_C = parm1[0]    
        
        modelsvr = regress_NuSVR(X_train, X_test, y_train, y_test, par_C, par_nu)
            
        matrix2[time,0] = modelsvr[0]  # R^2 result
        matrix2[time,1] = modelsvr[1]  # rmse result
        matrix2[time,2] = parm1[0]     # C parameter
        matrix2[time,3] = parm1[1]     # Nu paramter
        print('time', time)
    
    final = np.array(matrix1)
    np.savetxt("/Ginzburg_Landau_equation/Results-files4/R2_rmse/knn_measures_m%d.txt"%(j), final, fmt="%.14f", delimiter=",")

    final1 = np.array(matrix2)
    np.savetxt("/Ginzburg_Landau_equation/Results-files4/R2_rmse/svr_measures_m%d.txt"%(j), final1, fmt="%.14f", delimiter=",")
print('end')

