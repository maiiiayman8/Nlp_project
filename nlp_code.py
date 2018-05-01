# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:58:24 2018

@author: mai ayman
"""

import pandas as pd
import numpy as np
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.naive_bayes import GaussianNB
from  sklearn.svm import SVC
from  sklearn.ensemble import RandomForestClassifier 

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

 ## data analysis
train.head()

#y = train.pop("Survived")
#y.head()

train.drop(['Name','Embarked','Ticket','Cabin'],inplace=True,axis=1)
test.drop(['Name','Embarked','Ticket','Cabin'],inplace=True,axis=1)
train.head()

#to create dummy data columns from categorial ones
train=pd.get_dummies(train)
test=pd.get_dummies(test)
train.head()

train.isnull().sum()
##el7agat ele vales bet3tha b null f 12 freatures
test.isnull().sum()

##fill the null age with mean age and null fare  with mean fare

train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)

#modeing l
train.info()

feature=train.drop('Survived' , axis=1 )
label=train['Survived']
clf=DecisionTreeClassifier()
clf.fit(feature,label)

predication = clf.predict(test)
print(predication)

#for index in range(len(predication)):
    #if index==0 :
      #sub = pd.read_csv('mission.csv')  
        

sub=pd.DataFrame()
sub['PassengerId']=test['PassengerId']
sub['Survived']=clf.predict(test)
sub.to_csv('submission.csv',index=False)

sub = pd.read_csv('submission.csv')
sub.head()


#df_filtered = sub.query('Survived == 0')
#print(df_filtered)


#cross validation
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle= True ,random_state=0)

servive =train.pop('Survived')
servive.head()

#kNN algo 
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train, servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#score
round(np.mean(score)*100, 2)


#decision tree

clf=DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train, servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#accuracy score
round(np.mean(score)*100, 2)

#random forest classifer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf=RandomForestClassifier(n_estimators =13)
scoring = 'accuracy'
score = cross_val_score(clf, train , servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#accuracy score
round(np.mean(score)*100, 2)

#naive bayes algo
from sklearn.naive_bayes import GaussianNB

clf= GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train , servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#accuracy score
round(np.mean(score)*100, 2)

#svm algo  vector support machine
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train , servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#accuracy score
round(np.mean(score)*100, 2)

#regression algo 
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(clf, train , servive , cv =k_fold , n_jobs=1, scoring= scoring )
print(score)
#accuracy score
round(np.mean(score)*100, 2)
