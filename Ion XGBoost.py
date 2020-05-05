#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:03:53 2020

@author: feichang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

type(train.shape[0])

train_clean = train
"""
#take a look at the train
plt.figure(figsize=(5,5)); res = 1000
plt.plot(range(0,train.shape[0],res),train.signal[0::res])
for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
plt.show()

#take a look at the test
plt.figure(figsize=(10,5)); res = 1000
plt.plot(range(0,test.shape[0],res),test.signal[0::res])
plt.show()
"""
#first, fix the first drift in seg 2, 500000-600000

a = 500000
b = 600000

seg_1 = train.loc[train.index[a:b], 'signal'].values
time_1 = train.loc[train.index[a:b], 'time'].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(time_1.reshape(-1,1),seg_1)

train_clean.loc[train.index[a:b], 'signal'] = train_clean.signal[a:b].values - regressor.coef_*(train_clean.time.values[a:b] - 50)

"""
plt.figure(figsize=(5,5)); res = 1000
plt.plot(range(0,train_clean.shape[0],res),train.signal[0::res])
for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
plt.show()
"""

#then fix the polynomial drift
a = 0 
while a < 4500001:
    b = a + 500000
    seg_2 = train.loc[train.index[a:b], 'signal'].values
    time_2 = train.loc[train.index[a:b], 'time'].values
    
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 2)
    time_poly = poly_reg.fit_transform(time_2.reshape(-1,1))

    #define poly regressor
    lin_reg2 = LinearRegression()
    lin_reg2.fit(time_poly,seg_2)

    drift_0 = lin_reg2.predict(time_poly)[0]
    drift = lin_reg2.predict(time_poly) - drift_0

    train_clean.loc[train.index[a:b], 'signal'] = train_clean.signal[a:b].values - drift
    a += 500000

#now the signal data is clean, look: 
"""
res = 1000
plt.plot(range(0,train.shape[0],res),train.signal[0::res])
"""

"""
We also need to take the average current of the "phase" into consideration
"""

a = 0 
train_clean['Mean'] = 0.
train_clean['stdev'] = 0.
while a < 4500001:
    b = a + 500000
    avg = np.mean(train_clean.signal[a:b].values, dtype = 'float32')
    std = np.std(train_clean.signal[a:b].values, dtype = 'float32')
    train_clean.Mean[a:b].values.fill(avg) 
    train_clean.stdev[a:b].values.fill(std)
    a += 500000

"""
train_clean.head()
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_clean[['signal','Mean','stdev']], train_clean['open_channels'], test_size = 0.25, random_state = 0)
"""
from sklearn.tree import DecisionTreeClassifier
classifier_1 = DecisionTreeClassifier(random_state = 0, max_depth = 11, 
                                      min_samples_split = 32, min_samples_leaf = 5)
classifier_1.fit(X_train, y_train)
"""
from xgboost import XGBClassifier
classifier_1 = XGBClassifier()
classifier_1.fit(X_train, y_train)

prediction = classifier_1.predict(X_test)

from sklearn.metrics import f1_score
F1 = f1_score(y_test, prediction, average = 'macro')
print('F1 score:', F1)

"""
now it's very close. We are ready to move to the next step
plt.plot(prediction,'red')
plt.plot(train_clean['open_channels'])
"""
"""
now we clean the test data
first take a look
plt.plot(test['signal'])
"""

#every 100000 points, a phase, till 1000000

test_clean = test
a = 0

while a < 900001:
    b = a + 100000
    seg_1 = test.loc[train.index[a:b], 'signal'].values
    time_1 = test.loc[train.index[a:b], 'time'].values

    regressor_3 = LinearRegression()
    regressor_3.fit(time_1.reshape(-1,1),seg_1)

    drift_0 = regressor_3.predict(time_1.reshape(-1,1))[0]
    drift = regressor_3.predict(time_1.reshape(-1,1)) - drift_0
    
    test_clean.loc[train.index[a:b], 'signal'] = test_clean.signal[a:b].values - drift
    a += 100000
"""
take a look
plt.plot(test_clean['signal'])
"""

#1000000 to 1500000 a polynomial
a = 1000000
b = 1500000
seg_2 = test.loc[train.index[a:b], 'signal'].values
time_2 = test.loc[train.index[a:b], 'time'].values
    
poly_reg = PolynomialFeatures(degree = 2)
time_poly = poly_reg.fit_transform(time_2.reshape(-1,1))

lin_reg2 = LinearRegression()
lin_reg2.fit(time_poly,seg_2)

drift_0 = lin_reg2.predict(time_poly)[0]
drift = lin_reg2.predict(time_poly) - drift_0

test_clean.loc[test_clean.index[a:b], 'signal'] = test_clean.signal[a:b].values - drift + 0.25

"""
take a look
plt.figure(figsize=(20,5)); res = 1000
plt.plot(range(0,test_clean.shape[0],res),test_clean.signal[0::res])

plt.plot(test_clean['signal'])
plt.figure(figsize=(20,5)); res = 1000
plt.plot(pd.read_csv('test.csv')['signal'])
"""

a = 0 
test_clean['Mean'] = 0.
test_clean['stdev'] = 0.

while a < 1900001:
    b = a + 100000
    avg = np.mean(test_clean.signal[a:b].values, dtype = 'float32')
    std = np.std(test_clean.signal[a:b].values, dtype = 'float32')
    test_clean.Mean[a:b].values.fill(avg) 
    test_clean.stdev[a:b].values.fill(std)
    a += 100000

test_clean.head()
prediction = classifier_1.predict(test_clean[['signal','Mean','stdev']])

"""
Take a look at the mean
plt.plot(test_clean['Mean'])
plt.plot(train_clean['Mean'])
"""

"""
output = pd.DataFrame({'time': test_clean['time'], 'open_channels': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
"""
samplesubmission = pd.read_csv('sample_submission.csv', dtype={'time': 'str'})
samplesubmission.info()
output = pd.DataFrame({'time': samplesubmission.time, 'open_channels': prediction})
output.to_csv('submission.csv', index=False)

sub = pd.read_csv('submission.csv')
sub.info()
