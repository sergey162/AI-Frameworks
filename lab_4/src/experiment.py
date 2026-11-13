"""Общий протокол №1–3, предобработка деревьев и модели случайного леса."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lab_1.src.experiment import SEED, evaluate
from lab_2.src.experiment import load_tasks
from lab_3.src.experiment import make_pipeline as tree_pipeline


def baseline_params(task):
    """Фиксированный исходный лес; OOB включён только для диагностики train."""
    return dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
                max_features="sqrt" if task == "classification" else 1.0,
                bootstrap=True, max_samples=None, oob_score=True)


def sklearn_model(task, **params):
    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return cls(random_state=SEED, n_jobs=1, **params)


def make_pipeline(X, task, mode="raw", estimator=None):
    return tree_pipeline(X, task, mode, sklearn_model(task, **baseline_params(task))
                         if estimator is None else estimator)
