from disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
import numpy as np


seed = 0

X,Y=make_multilabel_classification(n_samples=32, n_features=10, random_state=seed) 
print(X,Y)
dn=DisturbingNeighbors(random_state=seed)

matrizFitX = np.matrix([X[i,:] for i in range(16)], dtype=int)
matrizFitY = np.matrix([Y[i,:] for i in range(16)], dtype=int)

dn.fit(matrizFitX,matrizFitY)


matrizPredict=np.matrix([X[j+16,:] for j in range(len(X)-16)], dtype=int)

dn.predict(matrizPredict)