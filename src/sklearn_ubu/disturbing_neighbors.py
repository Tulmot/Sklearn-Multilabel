from .homogeneous_ensemble import HomogeneousEnsemble
from .base_disturbing_neighbors import BaseDisturbingNeighbors
from sklearn.tree import DecisionTreeClassifier

class DisturbingNeighbors(HomogeneousEnsemble):

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
