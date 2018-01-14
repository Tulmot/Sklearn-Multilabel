from .homogeneous_ensemble import HomogeneousEnsemble
from .base_rotation_forest import BaseRotationForest
from sklearn.tree import DecisionTreeClassifier


class RotationForest(HomogeneousEnsemble):
    """A Rotation Forest.

    Rotation forest is a multi-label ensemble, this method generating
    classifier ensembles based on feature extraction. Create a training set for
    a base classifier, the set of features is randomly split into K subsets and
    PCA is applied to each subset. The rotations of the K axis take place to
    form the new features for a base classifier. The idea is to improve the
    accuracy and diversity within the set. The diversity is based on the
    extraction of features for each base classifier.

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

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 random_state=None,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _validate_estimator(self):
        self.base_estimator_ = BaseRotationForest(
                base_estimator=self.base_estimator)
