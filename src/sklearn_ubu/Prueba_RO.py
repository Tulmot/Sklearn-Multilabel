# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:41:02 2017

@author: Tubi
"""
from random_oracles import BaseRandomOracles
from disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

seed = 0

X, y = make_multilabel_classification(
        n_samples=50, n_features=10, random_state=seed)

ro = BaseRandomOracles(base_estimator=DecisionTreeClassifier(
                       random_state=seed),random_state=seed)

#ro = DisturbingNeighbors(
#    random_state=seed, base_estimator_=BaseRandomOracles(
#            random_state=seed, base_estimator=DecisionTreeClassifier(
#                random_state=seed)))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, train_size=0.5, random_state=seed)

ro.fit(X_train,y_train)

y_predict = ro.predict(X_test)
#print(y_predict)

y_predict_proba = ro.predict_proba(X_test)
print(y_predict_proba)