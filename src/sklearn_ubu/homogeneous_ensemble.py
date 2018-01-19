import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
from sklearn.ensemble.bagging import MAX_INT


class HomogeneousEnsemble(BaseEnsemble):
    """A Homogeneous Ensemble.

    Homogeneous Ensemble It is an abstract class that calls different
    ensembles, to which it passes the number of iterations that they must
    realized.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the ensemble is built.

    n_estimators : integer
        The number of estimators in the ensemble.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.
    """
    def __init__(self,
                 base_estimator=ExtraTreeClassifier(),
                 n_estimators=10,
                 random_state=None,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_params = estimator_params

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
        def fit_estimator(estimator):
            """Recibimos un estimador, que es un clasifcador, sobre el que
            hacemos fit, y devolvemos el clasificador entrenado."""
            estimator.fit(X, y)
            return estimator

        self._validate_estimator()
        self.estimators_ = []
        self.random_state = check_random_state(self.random_state)
        seeds = self.random_state.randint(MAX_INT, size=self.n_estimators)
        for i in range(self.n_estimators):
            self._make_estimator(append=True, random_state=seeds[i])
        self.estimators_ = list(map(fit_estimator, self.estimators_))

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

        def predict_classifiers(classifiers):
            """Recibimos unos clasificadores ya entrenados sobre los que
            hacemos predict y los devolvemos.
            """
            return classifiers.predict(X)

        def binarize_list(my_list):
            """Recorremos la lista de listas y pasamos la lista a otra función.
            """
            return list(map(binarize, my_list))

        def binarize(num):
            """ Con esto conseguimos que nuestra lista de listas pase a ser
            binaria.
            """
            if num >= self.n_estimators/2:
                return 1
            return 0
        predictions = list(map(predict_classifiers, self.estimators_))
        predictions = np.asarray(predictions)
        average = predictions.sum(axis=0)
        if average.ndim < 2:
            binarizada = list(map(binarize, average))
        else:
            binarizada = list(map(binarize_list, average))
        binarizada = np.asarray(binarizada)
        return binarizada

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
        def predict_proba_classifiers(classifier):
            """Recibimos unos clasificadores ya entrenados sobre los que
            hacemos predict_proba y los devolvemos.
            """
            return classifier.predict_proba(X)

        def divide_list(my_list):
            """Recorremos la lista de listas y pasamos la lista a otra
            función."""
            return list(map(calc_average, my_list))

        def calc_average(num):
            """ Hallamos el promedio.
            """
            return num / self.n_estimators
        predictions = list(map(predict_proba_classifiers, self.estimators_))
        predictions = np.asarray(predictions)
        average = predictions.sum(axis=0)
        divide = list(map(divide_list, average))
        divide = np.asarray(divide)
        return divide

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
