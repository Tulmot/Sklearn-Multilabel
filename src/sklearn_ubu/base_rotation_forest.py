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

    def split(self, X):
        tam = X.shape[1]
        list_features = np.arange(tam)
        self.random_state.shuffle(list_features)
        list_features = list(list_features)
        while(len(list_features) % self.n_groups != 0):
            random_feature = self.random_state.randint(0, tam)
            list_features.append(random_feature)
        list_features = np.asarray(list_features)
        divide=np.split(
            list_features, list_features.shape[0]/self.n_groups)
        def separe(rand_features):
            return X[:, rand_features]
        return list(map(separe,divide))
        

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
        def pca_transform(subX):
            pca = decomposition.PCA(random_state=self.random_state)
            pca.fit(subX)
            return (pca, pca.transform(subX))
        self.random_state = check_random_state(self.random_state)
        self._split_group = self.split(X)
        pcas_transform = list(map(pca_transform, self._split_group))
        tuples_pcas=list(map(pca_transform, self._split_group))
        self._pcas = list(map(lambda t: t[0], tuples_pcas))
        pcas_transform = list(map(lambda t: t[1], tuples_pcas))
        pcas_transform = np.concatenate((pcas_transform),axis=1)

