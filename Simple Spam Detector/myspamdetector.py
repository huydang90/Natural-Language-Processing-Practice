#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:54:14 2019

@author: dangngochuy
"""
#import libraries
from sklearn.naive_bayes import MultinomialNB
import numpy as np 
import pandas as pd


#load data
data = pd.read_csv("spambase.data").as_matrix()

#shuffle the data randomly into a different training and test set everytime
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

#create training set and test set/ already random
X_train = X[:-100, ]
Y_train = Y[:-100, ]
X_test = X[-100:, ]
Y_test = Y[-100:, ]

#create model 
model = MultinomialNB()
model.fit(X_train, Y_train)
print("Classification rate for NB: ", model.score(X_test, Y_test))

#try another model: AdaBoost

from sklearn.ensemble import AdaBoostClassifier

model2 = MultinomialNB()
model2.fit(X_train, Y_train)
print("Classification rate for AdaBoost: ", model2.score(X_test, Y_test))

#Reached 88% accuracy 