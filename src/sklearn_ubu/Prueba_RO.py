from random_oracles import RandomOracles
from base_random_oracles import BaseRandomOracles
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

seed = 0

X, y = make_multilabel_classification(
        n_samples=50, n_features=10, random_state=seed)

#ro = BaseRandomOracles(base_estimator=DecisionTreeClassifier(
#                       random_state=seed),random_state=seed)

ro = RandomOracles(
    random_state=seed, base_estimator_=BaseRandomOracles(
            random_state=seed, base_estimator=DecisionTreeClassifier(
                random_state=seed)))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, train_size=0.5, random_state=seed)

ro.fit(X_train,y_train)

y_predict = ro.predict(X_test)
#print(y_predict)

y_predict_proba = ro.predict_proba(X_test)
print(y_predict_proba)

dist = hamming_loss(y_test, y_predict)

print(dist)

scores = cross_val_score(ro, X, y, cv=3)
print(scores)