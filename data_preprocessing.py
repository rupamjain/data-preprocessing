# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:58:02 2019

@author: Rupam
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


#Handling Missing Data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN' , strategy='mean' ,axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#Encoding some fields
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
x_sclr=StandardScaler()
x_train=x_sclr.fit_transform(x_train)
x_test=x_sclr.transform(x_test)




