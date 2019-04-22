# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:11:11 2018

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  
On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, 
killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked 
the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there 
were not enough lifeboats for the passengers and crew. Although there was some 
element of luck involved in surviving the sinking, some groups of people were
 more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people 
were likely to survive. In particular, we ask you to apply the tools of machine 
learning to predict which passengers survived the tragedy.

@author: pallo
"""

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



#import Data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
original_train = pd.read_csv('train.csv')
original_test = pd.read_csv('test.csv')


### Explore the data 

#Age Breakdown
train['Age'].plot.hist(bins = 50)
plt.show()

# Survival based on sex 
pivot_table_sex = train.pivot_table(index = "Sex", values ="Survived")
pivot_table_sex.plot.bar()
plt.show()
# It looks as if females had a much higher chance of survival

# Survival based on age
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='green',bins=50)
died["Age"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()

# Survival based on Embarked 
pivot_table_emb = train.pivot_table(index = "Embarked", values ="Survived")
pivot_table_emb.plot.bar()
plt.show()

# Survival based on Pclass
pivot_table_class = train.pivot_table(index = "Pclass", values ="Survived")
pivot_table_class.plot.bar()
plt.show()

# Survival based on # of Parent/children
pivot_table_parch = train.pivot_table(index = "Parch", values ="Survived")
pivot_table_parch.plot.bar()
plt.show()

# Survival based on # of Sib/Sp
pivot_table_sibsp = train.pivot_table(index = "SibSp", values ="Survived")
pivot_table_sibsp.plot.bar()
plt.show()



### Feature engineering

# Drop cabin since there are 687 values missing
train = train.drop(['Cabin', 'Fare', 'Ticket'], axis=1)
test = test.drop(['Cabin', 'Fare', 'Ticket'], axis =1)


# Fill missing Embarked with S
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')


# Replace Embarked with S=0, C=1, Q=2
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# Replace male with 0 and female with 1
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1


# Fill missing age with the median age
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())


# Create new column of family size
train['FamTot'] = train['SibSp'] + train['Parch'] + 1
test['FamTot'] = test['SibSp'] + test['Parch'] + 1

def Alone(f):
    if f == 1:
        return 1
    elif f > 1:
        return 0
    
train['Alone'] = train['FamTot'].apply(Alone)
test['Alone'] = test['FamTot'].apply(Alone)

# Check to see relationship of survival and alone/ Shows that those not alone
# have better chance of survival
pivot_table_Alone = train.pivot_table(index = "Alone", values ="Survived")
pivot_table_Alone.plot.bar()
plt.show()

   
# Check to see relationship of survival and FamTot/ Family sizes <= 4
# had a better chance of survival. We create a new column of family size

pivot_table_FamTot = train.pivot_table(index = "FamTot", values ="Survived")
pivot_table_FamTot.plot.bar()
plt.show()

def FamSize(n):
    if n <= 4:
        return 1
    else:
        return 0
train['FamilySize'] = train['FamTot'].apply(FamSize)
test['FamilySize'] = test['FamTot'].apply(FamSize)

# Check the relationship of Family Size
pivot_table_FamilySize = train.pivot_table(index = "FamilySize", values ="Survived")
pivot_table_FamilySize.plot.bar()
plt.show()

train = train.drop(['SibSp', 'FamTot', 'Parch'], axis=1)
test = test.drop(['SibSp', 'FamTot', 'Parch'], axis=1)

# Titles

# Gather title of person
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

# Check survival based on titles
pivot_table_Title= train.pivot_table(index = "Title", values ="Survived")
pivot_table_Title.plot.bar()
plt.show()

# Rank of Titles based on chance of survival
Special = ['Countess', 'Lady', 'Sir' , 'Mlle', 'Ms', 'Mme'] #0
Ladies = ['Mrs', 'Miss'] #1
Rank = ['Dr', 'Major', 'Col', 'Master', ] #2
Men = ['Mr'] # 3
Dead = ['Capt', 'Rev', 'Jonkheer', 'Don'] #4

def Tit(p):
    if p in Special:
        return 0
    elif p in Ladies:
        return 1
    elif p in Rank:
        return 2
    elif p in Men:
        return 3
    elif p in Dead:
        return 4

train['Title'] = train['Title'].apply(Tit)
test['Title'] = test['Title'].apply(Tit)

# Check survival besed on title
pivot_table_Title2= train.pivot_table(index = "Title", values ="Survived")
pivot_table_Title2.plot.bar()
plt.show()

#drop name
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# Age

def Adult(a):
    if (a >18) & (a <65):
        return 1
    elif a <=18:
        return 0
    elif a>=65:
        return 2
        
train['Age'] = train['Age'].apply(Adult)
test['Age'] = test['Age'].apply(Adult)

# Check survival based on age group
pivot_table_Age2= train.pivot_table(index = "Age", values ="Survived")
pivot_table_Age2.plot.bar()
plt.show()

### Model

X = train.drop(['Survived', 'PassengerId'], axis =1 )
Y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Log Reg
lr = LogisticRegression()
lr.fit(X_train,y_train)

predictions_log = lr.predict(X_test)
log_accuracy = accuracy_score(y_test, predictions_log)
print('Log Reg Accuracy' , log_accuracy)

# KNN 
knnacc=[]
for i in range(1,20):
    clf=KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train,y_train)
    predictions_knn = clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, predictions_knn)
    knnacc.append(knn_accuracy)
print('Knn Accuracy' , max(knnacc), i)
    
# Decision Tree 
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
predictions_tree = decision_tree.predict(X_test)
tree_accuracy = accuracy_score(y_test,predictions_tree )
print('Tree Accuracy' ,tree_accuracy )

# Random forest 
rf =RandomForestClassifier(bootstrap=True, n_estimators=100,max_depth=4, random_state=0)
rf.fit(X_train,y_train)
predictions_forest = rf.predict(X_test)
forest_accuracy = accuracy_score(y_test,predictions_forest )
print('Random Forest Accuracy' ,forest_accuracy )

# Gradient boosting classifier 
gbc = GradientBoostingClassifier(n_estimators =200, max_depth = 4)
gbc.fit(X_train,y_train)
predictions_gbc = gbc.predict(X_test)
gbc_accuracy = accuracy_score(y_test,predictions_gbc )
print('Gradient Boosting Accuracy' ,gbc_accuracy )



# Predict the test data
test['Title'] = test['Title'].fillna(1)
test_X = test.drop(['PassengerId'], axis =1 )
test_predictions = rf.predict(test_X)

tit_predictions = pd.DataFrame(test_predictions, columns = ['Survived'])
predictionsCSV = pd.DataFrame(test[['PassengerId']], columns=['PassengerId'] )
predictionsCSV['Survived'] = test_predictions
predictionsCSV.to_csv('predict_titanic.csv', index=False)