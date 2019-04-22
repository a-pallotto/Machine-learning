# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:25:34 2019

    You are provided hourly rental data spanning two years. For this competition, 
the training set is comprised of the first 19 days of each month, while the test 
set is the 20th to the end of the month. You must predict the total count of 
bikes rented during each hour covered by the test set, using only information 
available prior to the rental period.

datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals

target = 'count' 
@author: pallo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



### Import data
train_original = pd.read_csv('train.csv')
train = pd.read_csv('train.csv') 
test_original = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')

# Check for missing values
train.isnull().values.any()
train.isnull().sum()
# Check data types
train.info()

### Data visualizations
import seaborn as sns


ax1 = sns.countplot(x="season", data=train)
plt.show()

ax2 = sns.countplot(x="holiday", data=train)
plt.show()

ax3 = sns.countplot(x="workingday", data=train)
plt.show()

ax4 = sns.countplot(x="weather", data=train)
plt.show()

### Feature engineering

# Split datetime into year, month, day, and hour using datetime
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour

# test
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour

ax5 = sns.countplot(x="hour", data=train)
plt.show()

# Create dummy variable for the year: 0 for 2011 and 1 for 2012
def change_year(x):
    if x == 2011:
        return 0
    else:
        return 1
train['year'] = train['year'].apply(change_year)

# test
test['year'] = test['year'].apply(change_year)

# We can now drop the datetime column
train = train.drop(['datetime'], axis = 1)
test = test.drop(['datetime'], axis = 1)

#Drop the registered/ casual columns (not given in test)
train = train.drop(['casual','registered'], axis = 1)

# Change Temperate to Range. min =.82, max=41 
train['temp'].min()
train['temp'].max()

def Temp(t):
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

train['temp']= train['temp'].apply(Temp)
test['temp']= test['temp'].apply(Temp)

# round columns with deicmals
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

### Prepare data
from sklearn.model_selection import train_test_split


X = train.drop(['count'], axis =1 )
Y = train['count']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

### Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


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


