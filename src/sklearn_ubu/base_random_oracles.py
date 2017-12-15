import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator


class BaseRandomOracles(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_oracles=3,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_oracles = n_oracles
        self.random_state = random_state
        self._rnd_oracles = None
        self._m_oracles = None

    def _calc_rnd_oracles(self, X):
        """Calculamos un array random para seleccionar unas instancias
        aleatorias"""
        return self.random_state.choice(
            X.shape[0], self.n_oracles, replace=False)

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

        def train(nearest_oracles):
            """Entrenamos cada uno de los oraculos """
            oracle_near = np.asarray(nearest_oracles).astype(bool)
            Xp = X[oracle_near, :]
            yp = y[oracle_near, :]
            return self.base_estimator.fit(Xp, yp)
        self.random_state = check_random_state(self.random_state)
        self._rnd_oracles = self._calc_rnd_oracles(X)
        self._m_oracles = self._oracles(X)
        m_oracle = self._nearest_oracle(X)
        self._classifiers_train = list(map(train, m_oracle.T))

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
        def list_predict(inst_oracles):
            """Predecimos cada una de las instancias con el oraculo
            correspondiente mas cercano"""
            oracle_near = inst_oracles[n_features:]
            instance = inst_oracles[:n_features]
            oracle_near = list(oracle_near)
            index_classifier = oracle_near.index(1)
            return self._classifiers_train[index_classifier].predict(
                [instance])[0]

        n_features = X.shape[1]
        m_oracle = np.concatenate((X, self._nearest_oracle(X)), axis=1)
        self._classifiers_prediction = list(map(list_predict, m_oracle))
        self._classifiers_prediction = np.asarray(self._classifiers_prediction)
        return self._classifiers_prediction

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

        def predict_prob(num):
            """Según el numero devolvemos una probabilidad o otra """
            if num == 1:
                return [[0., 1.]]
            else:
                return [[1., 0.]]

        def list_predict_proba(inst_oracles):
            """Predecimos la probabilidad de cada una de las instancias con el
            oraculo correspondiente mas cercano"""
            oracle_near = inst_oracles[n_features:]
            instance = inst_oracles[:n_features]
            oracle_near = list(oracle_near)
            index_classifier = oracle_near.index(1)
            prediction_proba = self._classifiers_train[
                index_classifier].predict_proba([instance])
            if((min(prediction_proba, key=(lambda x: len(x[0])))).shape[1]):
                call_predict = self._classifiers_train[
                    index_classifier].predict([instance])
                return list(np.asarray(list(map(
                        predict_prob, call_predict[0]))))
            else:
                return prediction_proba

        n_features = X.shape[1]
        m_oracle = np.concatenate((X, self._nearest_oracle(X)), axis=1)
        self._classifiers_prediction_proba = list(map(
            list_predict_proba, m_oracle))
        self._classifiers_prediction_proba = np.concatenate((
                self._classifiers_prediction_proba), axis=1)
        self._classifiers_prediction_proba = np.asarray(
            self._classifiers_prediction_proba)
        convert_array = lambda prob: np.asarray(prob)
        self._classifiers_prediction_proba = list(map(
                convert_array, self._classifiers_prediction_proba))
        return self._classifiers_prediction_proba
