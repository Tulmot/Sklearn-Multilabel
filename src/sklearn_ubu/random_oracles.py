from .homogeneous_ensemble import HomogeneousEnsemble
from .base_random_oracles import BaseRandomOracles
from sklearn.tree import DecisionTreeClassifier


class RandomOracles(HomogeneousEnsemble):
    """A Random Oracles.

    Random Oracles is a multi-label ensemble, each classifier in the set is
    replaced by a miniensemble of a pair of subclassifiers with an oracle to
    choose between them.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the ensemble is built.

    n_estimators : integer
        The number of estimators in the ensemble.

    n_oracles : integer
        The number of oracles in the ensemble.
        
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
    BaseRandomOracles
    
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
    >>> from sklearn_ubu.random_oracles import RandomOracles
    
    >>> clf = RandomOracles(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...
    array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  0.66666667,  0.        ,  0.        ,  0.        ])
    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 n_oracles=2,
                 random_state=None,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_oracles=n_oracles
        self.estimator_params = estimator_params

    def _validate_estimator(self):
        self.base_estimator_ = BaseRandomOracles(
                base_estimator=self.base_estimator,
                n_oracles=self.n_oracles)
