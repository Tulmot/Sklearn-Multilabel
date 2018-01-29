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
        
    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
        
    See also
    --------
    BaseRotationForest
    
    References
    ----------
    
    .. [1] Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). Rotation
           forest: A new classifier ensemble method. IEEE transactions on
           pattern analysis and machine intelligence, 28(10), 1619-1630.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn_ubu.rotation_forest import RotationForest
    
    >>> clf = RotationForest(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...
    array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  0.66666667,  0.        ,  0.        ,  0.        ])
    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 n_groups=3,
                 random_state=None,
                 per_samples=0.75,
                 per_samples_classes=0.8,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_groups = n_groups
        self.random_state = random_state
        self.per_samples = per_samples
        self.per_samples_classes = per_samples_classes
        self.estimator_params = estimator_params

    def _validate_estimator(self):
        self.base_estimator_ = BaseRotationForest(
                base_estimator=self.base_estimator,
                n_groups = self.n_groups,
                per_samples = self.per_samples,
                per_samples_classes = self.per_samples_classes)
