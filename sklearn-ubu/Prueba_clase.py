from disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import numpy as np

seed = 0

X,y=make_multilabel_classification(n_samples=32, n_features=10, random_state=seed) 
dn=DisturbingNeighbors(random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, 
                                                    random_state=seed)

X_train=np.matrix(np.array(X_train))
X_test=np.matrix(np.array(X_test))
y_train=np.matrix(np.array(y_train))
y_test=np.matrix(np.array(y_test))

dn.fit(X_train,y_train)

y_predict=dn.predict(X_test)


dist=hamming_loss(y_test, y_predict)

print(dist)

