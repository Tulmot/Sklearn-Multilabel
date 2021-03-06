import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import is_multilabel
from sklearn.ensemble.base import BaseEnsemble


class BaseRandomOracles(ClassifierMixin, BaseEnsemble):
    """A Base Random Oracles.

    BaseRandomOracles is a base classifier.

    Parameters
    ----------
    base_estimator_ : It is the classifier that we will use to train our data
        set, what it receives is either empty or an object, if it is empty by
        default the DecisionTreeClassifier is used.

    n_oracles : They are the oracles that we want to choose from the data
        set, by default if nothing happens, 3 are chosen.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.    
    
    See also
    --------
    RandomOracles
    
    References
    ----------
    
    .. [1] Kuncheva, L. I., & Rodriguez, J. J. (2007). Classifier ensembles
           with a random linear oracle. IEEE Transactions on Knowledge and Data
           Engineering, 19(4), 500-508.
           
    .. [2] Pardo, C., Diez, J. J. R., Díez-Pastor, J. F., & García-Osorio, C.
           I. (2011). Random Oracles for Regression Ensembles. Ensembles in
           Machine Learning Applications, 373, 181-199.
           
    .. [3] Rodríguez, J. J., Díez-Pastor, J. F., & García-Osorio, C. (2013,
           May). Random Oracle Ensembles for Imbalanced Data. In International
           Workshop on Multiple Classifier Systems (pp. 247-258). Springer,
           Berlin, Heidelberg.

    .. [4] Rodríguez, J., & Kuncheva, L. (2007). Naïve Bayes ensembles with a
           random oracle. Multiple Classifier Systems, 450-458.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn_ubu.base_random_oracles import BaseRandomOracles
    
    >>> clf = BaseRandomOracles(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...
    array([ 1.        ,  0.93333333,  1.        ,  0.93333333,  0.93333333,
        0.86666667,  0.93333333,  0.93333333,  1.        ,  1.        ])
    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_oracles=2,
                 random_state=None,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.base_estimator_ = base_estimator
        self.n_oracles = n_oracles
        self.random_state = random_state
        self.estimator_params = estimator_params
        """A random array of integers, the size of this array depends on the
        variable n_oracles, will select random rows of the data set, is what we
        will call random oracles."""
        self._rnd_oracles = None
        """It is a matrix that contains the instances of the random sentences
        that we have selected."""
        self._m_oracles = None

    def _calc_rnd_oracles(self, X):
        """Calculamos un array random para seleccionar unas instancias
        aleatorias"""
        return self.random_state.choice(
            X.shape[0], self.n_oracles, replace=False)

    def _oracles(self, X):
        """Calculamos la matriz de los oracles."""
        return X[self._rnd_oracles, :]

    def _nearest_oracle(self, _m_reduce):
        """Calculamos los oráculos más cercanos a las instancias escogidas
        antes aleatoriamente"""
        def euc_dis_func(t):
            return euclidean_distances([t], self._m_oracles).argmin()
        dn_nn = lambda inst: np.apply_along_axis(euc_dis_func, 1, [inst])
        m_nearest = np.apply_along_axis(dn_nn, 1, _m_reduce)
        oracles = lambda vec: np.concatenate((np.concatenate((
                                np.zeros(vec), [1]), axis=0), np.zeros(len(
                                    self._m_oracles)-vec-1)), axis=0)
        return np.apply_along_axis(oracles, 1, m_nearest)

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
            """Si no es multilabel"""
            if not self._multilabel:
                cls=self._make_estimator(append=False, random_state=self.random_state)
                return cls.fit(X, y)
            else:
                Xp = X[oracle_near, :]
                yp = y[oracle_near, :]
                cls=self._make_estimator(append=False, random_state=self.random_state)
                return cls.fit(Xp, yp)
        if is_multilabel(y):
            self._multilabel = True
        else:
            self._multilabel = False
        self.random_state = check_random_state(self.random_state)
        self._rnd_oracles = self._calc_rnd_oracles(X)
        self._m_oracles = self._oracles(X)
        self._classifiers_train = list(map(train, self._nearest_oracle(X).T))
        return self

    def _split_inst_oracles(self, X, inst_oracles):
        """Separamos la instancia y el oraculo, y de este último obtenemos cual
        es el oráculo más cercano a esa instancia """
        n_features = X.shape[1]
        oracle_near = inst_oracles[n_features:]
        instance = inst_oracles[:n_features]
        oracle_near = list(oracle_near)
        index_classifier = oracle_near.index(1)
        return instance, index_classifier

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
            """Predecimos cada una de las instancias con el oráculo
            correspondiente más cercano"""
            index_classifier = self._split_inst_oracles(X, inst_oracles)[1]
            instance = self._split_inst_oracles(X, inst_oracles)[0]
            return self._classifiers_train[index_classifier].predict(
                [instance])[0]
        m_oracle = np.concatenate((X, self._nearest_oracle(X)), axis=1)
        self._classifiers_prediction = list(map(list_predict, m_oracle))
        return np.asarray(self._classifiers_prediction)

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
            """Según el número devolvemos una probabilidad o otra """
            if num == 1:
                return [[0., 1.]]
            else:
                return [[1., 0.]]

        def list_predict_proba(inst_oracles):
            """Predecimos la probabilidad de cada una de las instancias con el
            oráculo correspondiente más cercano"""
            index_classifier = self._split_inst_oracles(X, inst_oracles)[1]
            instance = self._split_inst_oracles(X, inst_oracles)[0]
            prediction_proba = self._classifiers_train[
                index_classifier].predict_proba([instance])
            """Calculamos el array con el tamaño más pequeño, porque podemos
            tener el problema de que está considerando que la segunda salida
            solo tiene un valor. En este caso para no tener problemas lo que
            hacemos es calcular la probabilidad para esa instancia con el
            predict en vez de usar el predict_proba"""
            """Si es multilabel """
            if(self._multilabel is True and min(prediction_proba, key=(
                    lambda x: len(x[0]))).shape[1]):
                call_predict = self._classifiers_train[
                    index_classifier].predict([instance])
                return list(np.asarray(list(map(
                        predict_prob, call_predict[0]))))
            return prediction_proba[0]

        m_oracle = np.concatenate((X, self._nearest_oracle(X)), axis=1)
        self._classifiers_prediction_proba = list(map(
            list_predict_proba, m_oracle))
        if(self._multilabel is True):
            self._classifiers_prediction_proba = np.concatenate((
                    self._classifiers_prediction_proba), axis=1)
        self._classifiers_prediction_proba = np.asarray(
            self._classifiers_prediction_proba)
        return self._classifiers_prediction_proba
