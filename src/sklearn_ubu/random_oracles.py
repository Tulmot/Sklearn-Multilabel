from .homogeneous_ensemble import HomogeneousEnsemble
from .base_random_oracles import BaseRandomOracles
from sklearn.tree import DecisionTreeClassifier

class RandomOracles(HomogeneousEnsemble):
    """A Random Oracles.
    
    Random Oracles is a multi-label ensemble, each classifier in the set is
    replaced by a miniensemble of a pair of subclassifiers with an oracle to
    choose between them.
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
        self.base_estimator_ = BaseRandomOracles(
                base_estimator=self.base_estimator)