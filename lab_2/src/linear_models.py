"""Линейные модели: математика и оптимизация реализованы на NumPy."""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


def sigmoid(z):
    """Устойчивая сигмоида без переполнения exp на больших |z|."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def logistic_loss_gradient_hessian(theta, X, y, C, sample_weight):
    """Средняя взвешенная log-loss + L2; последний столбец X — константа.

    Нормировка совпадает со sklearn: sum(s_i * loss_i) + ||w||²/(2C).
    Здесь всё выражение делится на sum(s_i); свободный член не штрафуется.
    """
    z = X @ theta
    p = sigmoid(z)
    total = sample_weight.sum()
    penalty = np.full(len(theta), 1.0 / (C * total))
    penalty[-1] = 0.0
    loss = (sample_weight @ (np.logaddexp(0.0, z) - y * z)) / total
    loss += 0.5 * np.dot(penalty * theta, theta)
    gradient = X.T @ (sample_weight * (p - y)) / total + penalty * theta
    curvature = sample_weight * p * (1.0 - p) / total
    hessian = X.T @ (curvature[:, None] * X) + np.diag(penalty)
    return float(loss), gradient, hessian


class NumpyLogisticRegression(ClassifierMixin, BaseEstimator):
    """Бинарная логистическая регрессия, L2, метод Ньютона с поиском шага."""

    def __init__(self, C=1.0, class_weight=None, max_iter=100, tol=1e-9):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C должен быть положительным конечным числом")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("Поддерживаются class_weight=None и balanced")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter должен быть положительным целым")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol должен быть положительным конечным числом")
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_2d=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Поддерживаются ровно два класса")
        weight = np.ones(len(y))
        if self.class_weight == "balanced":
            weight = len(y) / (2.0 * np.bincount(encoded)[encoded])
        design = np.column_stack([X, np.ones(len(X))])
        theta = np.zeros(design.shape[1])
        self.loss_history_ = []
        self.converged_ = False
        for iteration in range(self.max_iter):
            loss, gradient, hessian = logistic_loss_gradient_hessian(
                theta, design, encoded, self.C, weight)
            self.loss_history_.append(loss)
            if np.max(np.abs(gradient)) <= self.tol:
                self.converged_ = True
                break
            try:
                direction = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            step = 1.0
            slope = np.dot(gradient, direction)
            # Условие Армихо обеспечивает убывание целевой функции.
            for _ in range(60):
                candidate = theta - step * direction
                z = design @ candidate
                candidate_loss = weight @ (np.logaddexp(0.0, z) - encoded * z) / weight.sum()
                candidate_loss += np.dot(candidate[:-1], candidate[:-1]) / (2 * self.C * weight.sum())
                if candidate_loss <= loss - 1e-4 * step * slope:
                    break
                step *= 0.5
            else:
                break
            theta = candidate
        self.n_iter_ = iteration + 1
        self.coef_ = theta[:-1][None, :]
        self.intercept_ = theta[-1:]
        self.objective_, gradient, _ = logistic_loss_gradient_hessian(
            theta, design, encoded, self.C, weight)
        self.gradient_norm_ = float(np.max(np.abs(gradient)))
        self.converged_ = self.gradient_norm_ <= self.tol
        if not self.converged_:
            warnings.warn("Метод Ньютона не достиг tol", ConvergenceWarning)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["coef_", "classes_"])
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return (X @ self.coef_.T).ravel() + self.intercept_[0]

    def predict_proba(self, X):
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        # При z=0 выбирается первый класс, как в sklearn.
        scores = self.decision_function(X)
        return self.classes_[(scores > 0.0).astype(int)]


class NumpyLinearRegression(RegressorMixin, BaseEstimator):
    """МНК при alpha=0 и Ridge при alpha>0; свободный член без штрафа."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha должен быть конечным неотрицательным числом")
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        if y.ndim != 1:
            raise ValueError("Поддерживается одна целевая переменная")
        x_mean, y_mean = X.mean(axis=0), y.mean()
        centered, target = X - x_mean, y - y_mean
        if self.alpha > 0:
            # Расширенная система реализует ||Xw-y||² + alpha*||w||².
            centered = np.vstack([centered, np.sqrt(self.alpha) * np.eye(X.shape[1])])
            target = np.concatenate([target, np.zeros(X.shape[1])])
        self.coef_, _, self.rank_, self.singular_ = np.linalg.lstsq(centered, target, rcond=None)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
