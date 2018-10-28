#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:47:29 2017

@author: badrawy
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def prep_data_4_1_time():
    file = pd.read_csv('data/ngp.csv')
    df= pd.DataFrame(list(reversed(file['price'])))
    input_t=4
    output_t=1
    cols=list()
    names=list()
    for i in range(input_t,0, -1):
        cols.append(df.shift(i))
        names.append(i)
    for i in range(0,output_t):
        cols.append(df.shift(-i))
        names.append(i+input_t+1)
    
    n_df=pd.concat(cols,axis=1)
    n_df.columns=names
    n_df.dropna(inplace=True)
    X=n_df[[4,3,2,1]]
    Y=n_df[5]
    ones=np.ones(shape=(X.shape[0],1))
    X=X/X.max()
    X=np.append(X, ones, axis=1)
    Y=np.asarray(Y)
    Y=Y/Y.max()

    factor=int(.80*X.shape[0])

    return split(X,Y,factor)

def prep_data_8_2_time():
    file = pd.read_csv('data/timesereis_8_2.csv')
    X=file[['0','1','2','3','4','5','6','7']]
    X=X/X.max()
    Y=file[['8','9']]
    Y=Y/Y.max()

    ones=np.ones(shape=(X.shape[0],1))
    X=np.append(X, ones, axis=1)
    Y=np.asarray(Y)
    factor=int(.80*X.shape[0])

    return split(X,Y,factor)

def split(X,Y,factor):
    X_train=X[:factor]
    Y_train=Y[:factor]
    X_test=X[factor:]
    Y_test=Y[factor:]
    return X_train,Y_train,X_test,Y_test
    



#X_train,Y_train,X_test,Y_test=prep_data_4_1_time()
X_train,Y_train,X_test,Y_test=prep_data_8_2_time()



clf=MLPRegressor(hidden_layer_sizes=(900,7), activation='relu',
             solver='adam',alpha=0.001, 
             batch_size=13, learning_rate='constant',learning_rate_init=0.001, 
             max_iter=1000, shuffle=True,random_state=5, tol=0.001, 
             early_stopping=True,
             verbose=True,
             validation_fraction=0.2
             )

#train the model
clf.fit(X_train,Y_train)
#score of the model
print(clf.score(X_test,Y_test));


#test the model
pred = clf.predict(X_test)
print(mean_squared_error(Y_test,pred))

    