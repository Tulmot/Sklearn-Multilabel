"""   """

# Author: Eduardo Tubilleja Calvo

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.utils import check_random_state


class DisturbingNeighbors:
    """A Disturbing Neighbors.

     Parameters
    ----------
    base_estimator : It is the classifier that we will use to train our data
        set, what it receives is either empty or an object, if it is empty by
        default the DecisionTreeClassifier is used.

    n_neighbors : They are the neighbors that we want to choose from the data
        set, by default if nothing happens, 10 are chosen.

    n_features : It is the size of the random sub-space, according to which
        the random features that we are going to use to train our
        classifier are chosen, by default it is 0.5, that is, half of the
        features are taken, if the value that is passed is greater than
        1, that number of features is taken.

    _rnd_dimensions : A Boolean random array, its size is equal to the number
        of features of the set, but then the number of TRUE values it contains
        will be equal to the value of the variable n_features, the TRUE values,
        indicate which features are chosen to evaluate the set.

    _rnd_neighbors : A random array of integers, the size of this array depends
        on the variable n_neighbors, will select random rows of the data set,
        is what we will call disturbing neighbors.

    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_neighbors=10,
                 n_features=0.5,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_neighbors = n_neighbors
        self.n_features = n_features
        self.random_state = random_state
        self._rnd_dimensions = None
        self._rnd_neighbors = None

    def _calculate_features(self, X):
        """Calculamos el numero de caracteristicas que usaremos"""
        if self.n_features < 1:
            return round(X.shape[1]*self.n_features)
        else:
            return self.n_features

    def _random_boolean(self):
        """Calculamos un array random boolean que es el que nos indicara que
        caracteristicas que valoraremos"""
        self._rnd_dimensions = self.random_state.randint(0, 2, self.n_features)
        return self._rnd_dimensions.astype(bool)

    def _random_array(self, X):
        """Calculamos un array random para seleccionar unas instancias
        aleatorias"""
        tam = X.shape[0]
        return self.random_state.choice(tam, self.n_neighbors, replace=False)

    def _reduce_data(self, X):
        """Reducimos los datos obtenidos a las caracteristicas que vamos a
        evaluar, que seran las que hemos obtenido segun el array random
        boolean"""
        return X[:, self._rnd_dimensions]

    def _nearest_neighbor(self, m_reduce):
        """Calculamos los vecinos mas cercanos a las instancias escogidas
        antes aleatoriamente"""
        m_neighbors = np.zeros((len(m_reduce), len(self._rnd_neighbors)))
        indice_ins = -1
        for instancia in m_reduce:
            dist = math.inf
            indice_ins += 1
            indice_vec = -1
            for j in self._rnd_neighbors:
                indice_vec += 1
                dist_tmp = euclidean_distances(instancia, m_reduce[j, :])
                if dist_tmp < dist:
                    dist = dist_tmp
                    a = indice_ins
                    b = indice_vec
            m_neighbors[a][b] = 1
        return m_neighbors

    def fit(self, X, Y):
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
        self.random_state = check_random_state(self.random_state)
        self.n_features = self._calculate_features(X)
        self._rnd_dimensions = self._random_boolean()
        self._rnd_neighbors = self._random_array(X)
        m_reduce = self._reduce_data(X)
        m_neighbors = self._nearest_neighbor(m_reduce)
        m_train = np.concatenate((X, m_neighbors), axis=1)
        return self.base_estimator.fit(m_train, Y)

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
        m_reduce = self._reduce_data(X)
        m_neighbors = self._nearest_neighbor(m_reduce)
        m_train = np.concatenate((X, m_neighbors), axis=1)
        return self.base_estimator.predict(m_train)

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
        m_reduce = self._reduce_data(X)
        m_neighbors = self._nearest_neighbor(m_reduce)
        m_train = np.concatenate((X, m_neighbors), axis=1)
        return self.base_estimator.predict_proba(m_train)
