import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn import decomposition
from sklearn.utils import resample


class BaseRotationForest(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_groups=3,
                 random_state=None,
                 n_muestras=0.75):
        self.base_estimator = base_estimator
        self.n_groups = n_groups
        self.random_state = random_state
        self.n_muestras = n_muestras
        self._rnd_features = None

    def _calc_rnd_features(self, X):
        """Calculamos un array random para seleccionar unas características
        aleatorias"""
        tam = X.shape[1]
        list_features = np.arange(tam)
        self.random_state.shuffle(list_features)
        list_features = list(list_features)
        while(len(list_features) % self.n_groups != 0):
            random_feature = self.random_state.randint(0, tam)
            list_features.append(random_feature)
        return np.asarray(list_features)
        
    def split(self, X):
        divide=np.split(
            self._rnd_features, self._rnd_features.shape[0]/self.n_groups)
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
        def sample(subX):
            return resample(subX, n_samples = round(
                    subX.shape[0] * self.n_muestras), random_state=self.random_state)
        def pca_fit(samples):
            pca = decomposition.PCA(random_state=self.random_state)
            return pca.fit(samples)
        def pca_transform(pos_subX):
            return self._pcas[pos_subX[0]].transform(pos_subX[1])
        self.random_state = check_random_state(self.random_state)
        self._rnd_features=self._calc_rnd_features(X)
        split_group = self.split(X)
        sample_group = list(map(sample, split_group))
        self._pcas=list(map(pca_fit,sample_group))
        tuple_pos_subX=list(zip(range(len(self._pcas)),split_group))
        print(tuple_pos_subX)
        pcas_transform=list(map(pca_transform, tuple_pos_subX))
        pcas_transform = np.concatenate((pcas_transform),axis=1)
        self.base_estimator.fit(pcas_transform,y)
        
    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability.
        Parameters
        ----------
        X : It's a matrix of form = [n_instances, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : It's a matrix of form = [n_class]
            The predicted classes.
        """
        def pca_transform(pos_subX):
            return self._pcas[pos_subX[0]].transform(pos_subX[1])
        split_group = self.split(X)
        tuple_pos_subX=list(zip(range(len(self._pcas)),split_group))
        pcas_transform=list(map(pca_transform, tuple_pos_subX))
        pcas_transform = np.concatenate((pcas_transform),axis=1)
        return self.base_estimator.predict(pcas_transform)
    
    def predict_proba(self, X):
        """The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble.
         Parameters
        ----------
        X : It's a matrix of form = [n_instances, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        p : It's a matrix of form = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        def pca_transform(pos_subX):
            return self._pcas[pos_subX[0]].transform(pos_subX[1])
        split_group = self.split(X)
        tuple_pos_subX=list(zip(range(len(self._pcas)),split_group))
        pcas_transform=list(map(pca_transform, tuple_pos_subX))
        pcas_transform = np.concatenate((pcas_transform),axis=1)
        return self.base_estimator.predict_proba(pcas_transform)
        