from disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
import numpy as np


seed = 0

X,Y=make_multilabel_classification(n_samples=32, n_features=10, random_state=seed) 
dn=DisturbingNeighbors(random_state=seed)

matrizFitX = np.matrix([X[i,:] for i in range(16)], dtype=int)
matrizFitY = np.matrix([Y[i,:] for i in range(16)], dtype=int)

dn.fit(matrizFitX,matrizFitY)


matrizPredict=np.matrix([X[j+16,:] for j in range(len(X)-16)], dtype=int)
y_true=np.matrix([Y[j+16,:] for j in range(len(Y)-16)], dtype=int)

y_predict=dn.predict(matrizPredict)


dist=hamming_loss(y_true, y_predict)

print(dist)

