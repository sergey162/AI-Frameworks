"""Математика, устойчивость и совместимость собственных линейных моделей."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from lab_2.src.experiment import load_tasks
from lab_2.src.linear_models import (NumpyLinearRegression, NumpyLogisticRegression,
                                     logistic_loss_gradient_hessian, sigmoid)


def test_logistic_derivatives_by_finite_differences():
    rng = np.random.default_rng(12)
    X = np.column_stack([rng.normal(size=(19, 3)), np.ones(19)])
    y = rng.integers(0, 2, 19)
    theta = rng.normal(size=4)
    weight = rng.uniform(0.2, 2, 19)
    _, gradient, hessian = logistic_loss_gradient_hessian(theta, X, y, 0.7, weight)
    eps = 1e-5
    numeric_gradient, numeric_hessian = [], []
    for axis in np.eye(4) * eps:
        plus = logistic_loss_gradient_hessian(theta + axis, X, y, 0.7, weight)
        minus = logistic_loss_gradient_hessian(theta - axis, X, y, 0.7, weight)
        numeric_gradient.append((plus[0] - minus[0]) / (2 * eps))
        numeric_hessian.append((plus[1] - minus[1]) / (2 * eps))
    np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-9)
    np.testing.assert_allclose(hessian, np.array(numeric_hessian).T, atol=1e-9)


@pytest.mark.parametrize("weight", [None, "balanced"])
@pytest.mark.parametrize("C", [0.1, 10.0])
def test_logistic_matches_sklearn(C, weight):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(250, 5))
    y = np.where(rng.random(250) < sigmoid(X[:, 0] * 2 - 1.5), "yes", "no")
    own = NumpyLogisticRegression(C=C, class_weight=weight).fit(X, y)
    ref = LogisticRegression(C=C, class_weight=weight, solver="newton-cholesky",
                             max_iter=100, tol=1e-10).fit(X, y)
    assert own.converged_
    assert np.max(np.diff(own.loss_history_)) <= 1e-12
    np.testing.assert_allclose(own.coef_, ref.coef_, atol=1e-7)
    np.testing.assert_allclose(own.predict_proba(X), ref.predict_proba(X), atol=1e-8)
    np.testing.assert_array_equal(own.predict(X), ref.predict(X))


@pytest.mark.parametrize("alpha", [0.0, 0.1, 100.0])
def test_linear_collinearity_and_intercept(alpha):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4))
    X = np.column_stack([X + 5, X[:, 0] + 5, np.ones(80)])
    y = 100 + 3 * X[:, 0] - X[:, 2] + rng.normal(size=80)
    own = NumpyLinearRegression(alpha=alpha).fit(X, y)
    ref = (LinearRegression() if alpha == 0 else Ridge(alpha=alpha, solver="svd")).fit(X, y)
    np.testing.assert_allclose(own.predict(X), ref.predict(X), atol=1e-9)
    assert abs(np.mean(own.predict(X) - y)) < 1e-10


def test_extreme_logits_and_validation():
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(sigmoid([-10000., 0., 10000.]), [0., 0.5, 1.])
    for cls in [NumpyLinearRegression, NumpyLogisticRegression]:
        with pytest.raises(NotFittedError):
            cls().predict([[0.]])
        fitted = cls().fit([[0.], [1.]], [0, 1])
        with pytest.raises(ValueError):
            fitted.predict([[0., 1.]])
    for params in [dict(C=0), dict(C=np.inf), dict(max_iter=0), dict(tol=0), dict(class_weight="bad")]:
        with pytest.raises(ValueError):
            NumpyLogisticRegression(**params).fit([[0.], [1.]], [0, 1])
    with pytest.raises(ValueError):
        NumpyLinearRegression(alpha=-1).fit([[0.], [1.]], [0., 1.])
    for y in [[1, 1], [0.2, 0.7], [0, 1, 2]]:
        with pytest.raises(ValueError):
            NumpyLogisticRegression().fit(np.arange(len(y))[:, None], y)
    with pytest.warns(ConvergenceWarning):
        model = NumpyLogisticRegression(max_iter=1, tol=1e-15).fit([[0.], [1.]], [0, 1])
    assert not model.converged_


def test_common_data_protocol():
    tasks = load_tasks(Path(__file__).resolve().parents[2])
    assert len(tasks["classification"]["X_train"]) == 36000
    assert len(tasks["classification"]["X_test"]) == 9000
    assert len(tasks["regression"]["X_train"]) == 799
    assert len(tasks["regression"]["X_test"]) == 206
    # Полнота, контрольные суммы и непересечение рецептур проверяются в load_tasks.
    assert all(len(task["cv"]) == 3 for task in tasks.values())
