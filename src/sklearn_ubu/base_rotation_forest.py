import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn import decomposition
from sklearn.utils import resample
from sklearn.utils.multiclass import is_multilabel


class BaseRotationForest(ClassifierMixin, BaseEstimator):
    """A Base Rotation Forest.

    BaseRotationForest is a base classifier.

    Parameters
    ----------
    base_estimator_ : It is the classifier that we will use to train our data
        set, what it receives is either empty or an object, if it is empty by
        default the DecisionTreeClassifier is used.

    n_groups : They are the groups that we want split our data set, by default
    if is none, 3 are chosen.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    per_samples : They size of the sample of each of the subsets, by default
    if is none, 75% are chosen.

    per_samples_classes : They size of the classes of each of the subsets, by
    default if is none, 80% are chosen.

    _rnd_features : A random array of integers, that will be used to split the
    set.
    """

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_groups=3,
                 random_state=None,
                 per_samples=0.75,
                 per_samples_classes=0.8):
        self.base_estimator = base_estimator
        self.n_groups = n_groups
        self.random_state = random_state
        self.per_samples = per_samples
        self.per_samples_classes = per_samples_classes
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

    def _split(self, X):
        """Dividimos el conjunto de datos, por defecto cada subgrupo tendra
        tamaño 3"""
        divide = np.split(
            self._rnd_features, self._rnd_features.shape[0]/self.n_groups)

        def separe(rand_features):
            return X[:, rand_features]
        return list(map(separe, divide))

    def _pca_transform(self, pos_subX):
        """Hacemos la transformación de cada subgrupo con su respectivo pca"""
        return self._pcas[pos_subX[0]].transform(pos_subX[1])

    def _split_transform(self, X):
        """Dividimos el conjunto, tranformamos los subgrupos y los concatenamos
        para obtener el conjunto final"""
        split_group = self._split(X)
        tuple_pos_subX = list(zip(range(len(self._pcas)), split_group))
        sub_pcas_transform = list(map(self._pca_transform, tuple_pos_subX))
        sub_pcas_transform = np.concatenate((sub_pcas_transform), axis=1)
        return sub_pcas_transform

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
        def get_sample(subX):
            """Obtenemos una muestra ddel subconjunto"""
            return resample(subX, replace=False, n_samples=round(
                    subX.shape[0]*self.per_samples
                    ), random_state=self.random_state)

        def get_sample_class(suby):
            """Obtenemos una muestra ddel subconjunto"""
            return resample(suby, replace=False, n_samples=round(
                    suby.shape[0]*self.per_samples_classes
                    ), random_state=self.random_state)

        def pca_fit(samples):
            """Entrenamos pca para cada una de las muestras"""
            pca = decomposition.PCA(random_state=self.random_state)
            return pca.fit(samples)

        def get_instance(sample_class):
            """Obtenemos una lista con las distintas instancias de X según las
            clases"""
            def get_pos(pos):
                """Obtenemos las instancias de X que corresponden con la
                clase"""
                if (sample_class == y[pos]).all():
                    return X[pos]
            return list(filter(None.__ne__, map(
                get_pos, np.arange(y.shape[0]))))

        self.random_state = check_random_state(self.random_state)
        self._rnd_features = self._calc_rnd_features(X)
        classes = []
        instances_classes = []
        self.list_classes_X = []
        """Si es multilabel o si es singlelabel"""
        if is_multilabel(y):
            """Obtengo las distintas clases"""
            classes = np.asarray(list({tuple(x) for x in y}))
        else:
            classes = np.asarray(list(set(y)))
        """Me quedo con una parte de las distintas clases"""
        samples_classes = get_sample_class(classes)
        instances_classes = np.concatenate(
            list(map(get_instance, samples_classes)))

        instances_classes = np.asarray(instances_classes)
        split_group = self._split(X)
        sample_group = list(map(get_sample, self._split(instances_classes)))
        self._pcas = list(map(pca_fit, sample_group))
        tuple_pos_subX = list(zip(range(len(self._pcas)), split_group))
        sub_pcas_transform = list(map(self._pca_transform, tuple_pos_subX))
        sub_pcas_transform = np.concatenate((sub_pcas_transform), axis=1)
        return self.base_estimator.fit(sub_pcas_transform, y)

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
        return self.base_estimator.predict(self._split_transform(X))

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
        return self.base_estimator.predict_proba(self._split_transform(X))
