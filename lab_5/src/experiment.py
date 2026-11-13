"""Общий протокол №1–4 и Pipeline градиентного бустинга."""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lab_1.src.experiment import SEED, evaluate
from lab_2.src.experiment import load_tasks
from lab_3.src.experiment import make_pipeline as tree_pipeline


def baseline_params(task):
    """Параметры фиксируются до подбора, одинаковая сложность для двух задач."""
    return dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                min_samples_leaf=1, subsample=1.0)


def sklearn_model(task, **params):
    cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
    # Friedman MSE и обычный SSE ранжируют пороги одинаково для скалярных остатков.
    return cls(random_state=SEED, n_iter_no_change=None,
               loss="log_loss" if task == "classification" else "squared_error", **params)


def make_pipeline(X, task, mode="raw", estimator=None):
    return tree_pipeline(X, task, mode, sklearn_model(task, **baseline_params(task))
                         if estimator is None else estimator)
