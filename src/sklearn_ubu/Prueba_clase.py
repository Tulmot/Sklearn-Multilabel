from disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

seed = 0

X,y=make_multilabel_classification(n_samples=32, n_features=10, random_state=seed) 
bc=OneVsRestClassifier(BaggingClassifier())
#dn=BaggingClassifier(base_estimator=DisturbingNeighbors(random_state=seed))
dn=DisturbingNeighbors(base_estimator=DecisionTreeClassifier(max_depth=3),random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, 
                                                    random_state=seed)




train=dn.fit(X_train,y_train)
#train2=bc.fit(X_train,y_train)
y_predict=dn.predict(X_test)
y_predict_proba=dn.predict_proba(X_test)

dist=hamming_loss(y_test, y_predict)

scores = cross_val_score(dn, X, y)
print(scores)

print(dist)




