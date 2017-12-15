import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn import decomposition

class BaseRotationForest(ClassifierMixin, BaseEstimator):
    
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_groups=3,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_groups = n_groups
        self.random_state = random_state
        
    def split(self,X):
        tam = X.shape[0]
        list_instances=np.arange(tam)
        self.random_state.shuffle(list_instances)
        list_instances=list(list_instances)
        while(len(list_instances)%self.n_groups!=0):
            list_instances.append(self.random_state.randint(0,tam))
        list_instances=np.asarray(list_instances)
        return X[np.split(list_instances, list_instances.shape[0]/self.n_groups),:]
        
    def fit(self, X, y):
        """Build a Bagging ensemble of estimators from the training set (X, y).
        Parameters
        ----------
        X : It's a matrix of form = [n_instances, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : It's a matrix of form = [n_class]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        def pca_fit_transform(subX):
            pca.fit(subX)
            return pca.transform(subX)
        self.random_state = check_random_state(self.random_state)
        self._split_group=self.split(X)
        pca = decomposition.PCA()
        pcas=list(map(pca_fit_transform,self._split_group))
        print(X)
        print(np.concatenate(pcas))
        pcas=np.concatenate(pcas)
        print(self.base_estimator.fit(pcas, y))