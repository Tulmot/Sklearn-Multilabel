from .homogeneous_ensemble import HomogeneousEnsemble
from .base_rotation_forest import BaseRotationForest
from sklearn.tree import DecisionTreeClassifier

class RotationForest(HomogeneousEnsemble):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=3,
                 random_state=None,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _validate_estimator(self):
        self.base_estimator_ = BaseRotationForest(
                base_estimator=self.base_estimator)
