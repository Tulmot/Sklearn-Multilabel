from .homogeneous_ensemble import HomogeneousEnsemble
from .base_disturbing_neighbors import BaseDisturbingNeighbors
from sklearn.tree import DecisionTreeClassifier


class DisturbingNeighbors(HomogeneousEnsemble):
    """A Disturbing Neighbors.

    Disturbing neighbors is a multi-label ensemble, this method alters the
    normal training process of the base classifiers in an ensemble, improving
    their diversity and accuracy. DN creates new features using a 1-NN
    classifier, these characteristics are the 1-NN output plus a set of Boolean
    attributes indicated by the nearest neighbor.

    The 1-NN classifier is created using a small subset of training instances,
    chosen at random from the original set. The dimensions to calculate the
    Euclidean distance are also random. With these characteristics created when
    we train base classifiers with them, diversity increases.

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
        
    See also
    --------
    BaseDisturbingNeighbors
    
    References
    ----------
    
    .. [1] Maudes, J., Rodríguez, J., & García-Osorio, C. (2009). Disturbing
           neighbors diversity for decision forests. Applications of Supervised
           and Unsupervised Ensemble Methods, 113-133.
           
    .. [2] Maudes, J., Rodríguez, J. J., & García-Osorio, C. I. (2009, June).
           Disturbing Neighbors Ensembles for Linear SVM. In MCS (pp. 191-200).
           
    .. [3] Pardo, C., Rodríguez, J. J., García-Osorio, C., & Maudes, J. (2010,
       June). An empirical study of multilayer perceptron ensembles for 
       regression tasks. In International Conference on Industrial, Engineering
       and Other Applications of Applied Intelligent Systems (pp. 106-115).
       Springer, Berlin, Heidelberg.
           
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn_ubu.disturbing_neighbors import DisturbingNeighbors
    
    >>> clf = DisturbingNeighbors(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  0.66666667,  0.        ,  0.        ,  0.        ])
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
        self.base_estimator_ = BaseDisturbingNeighbors(
                base_estimator=self.base_estimator)
