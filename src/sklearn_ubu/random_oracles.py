# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:39:23 2017

@author: Tubi
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator


class BaseRandomOracles(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_oracles=10,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_oracles = n_oracles
        self.random_state = random_state
        self._rnd_oracles = None
        self._m_oracles = None
        
    def _calc_rnd_oracles(self, X):
        """Calculamos un array random para seleccionar unas instancias
        aleatorias"""
        tam = X.shape[0]
        return self.random_state.choice(tam, self.n_oracles, replace=False)
    
    def _oracles(self, X):
        """Calculamos la matriz de los oracles."""
        self._m_oracles = X[self._rnd_oracles, :]
        return self._m_oracles
    
    def _nearest_oracle(self, _m_reduce):
        """Calculamos los vecinos mas cercanos a las instancias escogidas
        antes aleatoriamente"""
        def euc_dis_func(t):
            return euclidean_distances([t], self._m_oracles).argmin()
        dn_nn = lambda inst: np.apply_along_axis(euc_dis_func, 1, [inst])
        m_nearest = np.apply_along_axis(dn_nn, 1, _m_reduce)
        oracles = lambda vec: np.concatenate((np.concatenate((
                                np.zeros(vec), [1]), axis=0), np.zeros(len(
                                    self._m_oracles)-vec-1)), axis=0)
        m_oracles = np.apply_along_axis(oracles, 1, m_nearest)
        return m_oracles
    
    def fit(self, X):
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
        self._rnd_oracles = self._calc_rnd_oracles(X)
        self._m_oracles = self._oracles(X)
        m_oracle = self._nearest_oracle(X)
        m_train = np.concatenate((X, m_oracle), axis=1)
        return self.base_estimator.fit(m_train)