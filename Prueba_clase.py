from sklearn.disturbing_neighbors import DisturbingNeighbors
from sklearn.datasets import make_multilabel_classification
import numpy as np
from sklearn.utils import check_random_state

random_state=None
random_state = check_random_state(random_state)

seed = 0

X,Y=make_multilabel_classification(n_samples=12, n_features=10, random_state=seed) 

dn=DisturbingNeighbors(random_state=seed)

matrizFitX = np.matrix([X[i,:] for i in range(5)], dtype=int)
matrizFitY = np.matrix([Y[i,:] for i in range(5)], dtype=int)

dn.fit(matrizFitX,matrizFitY)


matrizPredict=np.matrix([X[j+5,:] for j in range(len(X)-5)], dtype=int)

dn.predict(matrizPredict)