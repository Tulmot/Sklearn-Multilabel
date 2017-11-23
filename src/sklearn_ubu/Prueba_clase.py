from disturbing_neighbors import DisturbingNeighbors
from disturbing_neighbors import BaseDisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
# from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


seed = 0

X, y = make_multilabel_classification(
        n_samples=80, n_features=10, random_state=seed)

# bc=OneVsRestClassifier(BaggingClassifier())
# dn=BaggingClassifier(base_estimator=DisturbingNeighbors(random_state=seed))
# dn=BaseDisturbingNeighbors(base_estimator=DecisionTreeClassifier(
#                       random_state=seed),random_state=seed)
dn = DisturbingNeighbors(
    random_state=seed, base_estimator_=BaseDisturbingNeighbors(
            random_state=seed, base_estimator=DecisionTreeClassifier(
                random_state=seed)))
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, train_size=0.5, random_state=seed)


clas_train = dn.fit(X_train, y_train)

# train2=bc.fit(X_train,y_train)
y_predict = dn.predict(X_test)
# print(y_predict)
y_predict_proba = dn.predict_proba(X_test)
# print(y_predict_proba)

dist = hamming_loss(y_test, y_predict)

print(dist)

# scores = cross_val_score(dn, X, y, cv=5)
# print(scores)
