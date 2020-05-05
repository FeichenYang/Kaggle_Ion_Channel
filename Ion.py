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
res = 1000

plt.plot(range(0,train.shape[0],res),train.signal[0::res])

"""
I tried to go straight ot a random forest. It doesn't work.
We also need to take the average current of the "phase" into consideration

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 3, random_state = 0)
classifier.fit(train_clean.loc[:,'signal'].values.reshape(-1,1),train_clean.open_channels[:])
"""

"""
I tried to go straight ot a random forest. It doesn't work.
We also need to take the average current of the "phase" into consideration
"""

a = 0 
train_clean['Mean'] = 0.

while a < 4500001:
    b = a + 500000
    avg = np.mean(train_clean.signal[a:b].values, dtype = 'float32')
    train_clean.Mean[a:b].values.fill(avg) 
    a += 500000

"""
train_clean.head()
"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(train_clean[['signal','Mean']],train_clean['open_channels'])

prediction = classifier.predict(train_clean[['signal','Mean']])

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

test_clean.loc[test_clean.index[a:b], 'signal'] = test_clean.signal[a:b].values - drift

"""
take a look
plt.plot(test_clean['signal'])
"""

a = 0 
test_clean['Mean'] = 0.

while a < 1900001:
    b = a + 100000
    avg = np.mean(test_clean.signal[a:b].values, dtype = 'float32')
    test_clean.Mean[a:b].values.fill(avg) 
    a += 100000

test_clean.head()
prediction = classifier.predict(test_clean[['signal','Mean']])
"""
output = pd.DataFrame({'time': test_clean['time'], 'open_channels': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
"""
samplesubmission = pd.read_csv('sample_submission.csv')

output = pd.DataFrame({'time': test_clean.time, 'open_channels': prediction})
output.to_csv('submission.csv', index=False)

sub = pd.read_csv('submission.csv')
sub.head()