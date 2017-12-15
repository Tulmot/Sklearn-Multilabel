from rotation_forest import RotationForest
from base_rotation_forest import BaseRotationForest
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


seed = 0

X, y = make_multilabel_classification(
        n_samples=50, n_features=10, random_state=seed)

rf=BaseRotationForest(base_estimator=DecisionTreeClassifier(
                      random_state=seed),random_state=seed)
#dn = DisturbingNeighbors(
#    random_state=seed, base_estimator_=BaseDisturbingNeighbors(
#            random_state=seed, base_estimator=DecisionTreeClassifier(
#                random_state=seed)))
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, train_size=0.5, random_state=seed)

clas_train = rf.fit(X_train, y_train)