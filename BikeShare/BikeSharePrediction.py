# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:25:34 2019
 
@author: pallo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


# Import data
train_original = pd.read_csv('train.csv')
train = pd.read_csv('train.csv') 
test_original = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')

# Check for missing values
train.isnull().values.any()
train.isnull().sum()
# Check data types
train.info()

# Data visualizations
ax1 = sns.countplot(x="season", data=train)
plt.show()

ax2 = sns.countplot(x="holiday", data=train)
plt.show()

ax3 = sns.countplot(x="workingday", data=train)
plt.show()

ax4 = sns.countplot(x="weather", data=train)
plt.show()

# Feature engineering

# Split datetime into year, month, day, and hour using datetime
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour

# Do same for test data
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour

ax5 = sns.countplot(x="hour", data=train)
plt.show()


def change_year(x):
    '''
    Create dummy variable for the year: 0 for 2011 and 1 for 2012
    '''
    if x == 2011:
        return 0
    else:
        return 1
train['year'] = train['year'].apply(change_year)

# test
test['year'] = test['year'].apply(change_year)

# Drop the datetime column
train = train.drop(['datetime'], axis = 1)
test = test.drop(['datetime'], axis = 1)

# Drop the registered/ casual columns (not given in test)
train = train.drop(['casual', 'registered'], axis = 1)

# Change Temperate to Range. min =.82, max=41 
train['temp'].min()
train['temp'].max()

def temp_function(t):
    '''
    Function to create dummy variables for temperature
    '''
    if (t>0) & (t<5):
        return 0
    if (t>=5) & (t<10):
        return 1
    if (t>=10) & (t<15):
        return 2
    if (t>=15) & (t<20):
        return 3
    if (t>=20) & (t<25):
        return 4
    if (t>=25) & (t<30):
        return 5
    if (t>=30) & (t<35):
        return 6
    if (t>=35) & (t<40):
        return 7
    if (t>=40) & (t<55):
        return 8

train['temp']= train['temp'].apply(temp_function)
test['temp']= test['temp'].apply(temp_function)

# Round columns contatining deicmals
train.atemp = train.atemp.round()
train.windspeed = train.windspeed.round()

test.windspeed = test.windspeed.round()

# Create new columns to seperate seasons
train['spring'] = np.where(train['season']==1, 1, 0)
train['summer'] = np.where(train['season']==2, 1, 0)
train['fall'] = np.where(train['season']==3, 1, 0)
train['winter'] = np.where(train['season']==4, 1, 0)

train = train.drop(['season'], axis=1)

test['spring'] = np.where(test['season']==1, 1, 0)
test['summer'] = np.where(test['season']==2, 1, 0)
test['fall'] = np.where(test['season']==3, 1, 0)
test['winter'] = np.where(test['season']==4, 1, 0)

test = test.drop(['season'], axis=1)

# Create new columns to seperate weather types
train['clear'] = np.where(train['weather']==1, 1, 0)
train['light'] = np.where(train['weather']==2, 1, 0)
train['moderate'] = np.where(train['weather']==3, 1, 0)
train['heavy'] = np.where(train['weather']==4, 1, 0)

train = train.drop(['weather'], axis=1)

test['clear'] = np.where(test['weather']==1, 1, 0)
test['light'] = np.where(test['weather']==2, 1, 0)
test['moderate'] = np.where(test['weather']==3, 1, 0)
test['heavy'] = np.where(test['weather']==4, 1, 0)

test = test.drop(['weather'], axis=1)

# Drop 'feels like temp'
train = train.drop(['atemp'],axis =1)
test = test.drop(['atemp'],axis =1)

# Create a correlation matrix
cor_mat = train.corr()
sns.heatmap(data = cor_mat,annot = True)

# Prepare data
X = train.drop(['count'], axis =1 )
Y = train['count']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

### Model
rfr = RandomForestRegressor(n_estimators = 500, n_jobs=-1)
rfr.fit(X_train,y_train)
predictions_rfr = rfr.predict(X_test)

# RMSLE
accuracy = np.sqrt(mean_squared_log_error(y_test, predictions_rfr))
print('Random Forest Accuracy' , accuracy)

# Create predictions 
final_pred = rfr.predict(test)
d = {'datetime':test_original['datetime'], 'count': final_pred}
solution = pd.DataFrame(d)
solution.to_csv('solution.csv', index=False)


