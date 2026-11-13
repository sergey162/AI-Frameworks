"""CART на NumPy: полный перебор порогов, Gini и squared_error.

sklearn используется только для интерфейса estimator и проверки входов.
Поиск разбиений, рост дерева, вероятности и прогноз реализованы самостоятельно.
"""

from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class _CART(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _validate_parameters(self):
        for name, lower in [("max_depth", 1), ("min_samples_split", 2), ("min_samples_leaf", 1)]:
            value = getattr(self, name)
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
                raise ValueError(f"{name} должен быть целым числом >= {lower}")
        if not np.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease должен быть конечным и >= 0")

    def _fit(self, X, y, classification):
        self._validate_parameters()
        X, y = validate_data(self, X, y, dtype=np.float32, y_numeric=not classification)
        if y.ndim != 1:
            raise ValueError("Поддерживается одна целевая переменная")
        self._classification = classification
        if classification:
            check_classification_targets(y)
            self.classes_, target = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            if self.class_weight not in (None, "balanced"):
                raise ValueError("Поддерживаются class_weight=None и balanced")
            counts = np.bincount(target, minlength=self.n_classes_)
            weights = np.ones(len(y)) if self.class_weight is None else len(y) / (self.n_classes_ * counts[target])
        else:
            # Центрирование уменьшает потерю точности при вычислении SSE.
            self.target_offset_ = float(np.mean(y))
            target = np.asarray(y, dtype=np.float64) - self.target_offset_
            weights = np.ones(len(y))
        self.nodes_ = []
        self.feature_importances_ = np.zeros(X.shape[1])
        self.max_depth_ = 0
        self.n_leaves_ = 0
        total_weight = weights.sum()
        # Стек вместо рекурсии позволяет строить и глубокие деревья.
        pending = [(np.arange(len(y)), 0, None, None)]
        while pending:
            indices, depth, parent, side = pending.pop()
            values, w = target[indices], weights[indices]
            mass = w.sum()
            if classification:
                histogram = np.bincount(values, weights=w, minlength=self.n_classes_)
                value = histogram / mass
                impurity = max(0.0, 1.0 - value @ value)
                parent_cost = mass * impurity
            else:
                mean = values.mean()
                value = mean + self.target_offset_
                parent_cost = float(np.sum((values - mean) ** 2))
                impurity = parent_cost / mass
            node = dict(feature=-1, threshold=np.nan, left=-1, right=-1, value=value,
                        n_samples=len(indices), impurity=impurity, depth=depth)
            node_id = len(self.nodes_)
            self.nodes_.append(node)
            if parent is not None:
                self.nodes_[parent][side] = node_id
            self.max_depth_ = max(self.max_depth_, depth)
            best = None
            best_cost = np.inf
            can_split = (len(indices) >= max(self.min_samples_split, 2 * self.min_samples_leaf)
                         and (self.max_depth is None or depth < self.max_depth)
                         and impurity > np.finfo(float).eps)
            if can_split:
                for feature in range(X.shape[1]):
                    order = np.argsort(X[indices, feature], kind="stable")
                    sorted_indices = indices[order]
                    x = X[sorted_indices, feature].astype(np.float64)
                    positions = np.arange(1, len(x))
                    # Почти равные float32-значения не образуют отдельный порог.
                    valid = ((x[1:] > x[:-1] + 1e-7)
                             & (positions >= self.min_samples_leaf)
                             & (len(x) - positions >= self.min_samples_leaf))
                    if not valid.any():
                        continue
                    ordered_target, ordered_w = target[sorted_indices], weights[sorted_indices]
                    if classification:
                        onehot = np.zeros((len(x), self.n_classes_))
                        onehot[np.arange(len(x)), ordered_target] = ordered_w
                        left_counts = np.cumsum(onehot, axis=0)[:-1]
                        right_counts = histogram - left_counts
                        left_mass = np.cumsum(ordered_w)[:-1]
                        right_mass = mass - left_mass
                        costs = (mass - (left_counts ** 2).sum(axis=1) / left_mass
                                 - (right_counts ** 2).sum(axis=1) / right_mass)
                    else:
                        cumulative = np.cumsum(ordered_target)[:-1]
                        cumulative2 = np.cumsum(ordered_target ** 2)[:-1]
                        total, total2 = ordered_target.sum(), np.sum(ordered_target ** 2)
                        costs = (cumulative2 - cumulative ** 2 / positions
                                 + total2 - cumulative2 - (total - cumulative) ** 2 / (len(x) - positions))
                    costs[~valid] = np.inf
                    cut = int(np.argmin(costs))
                    # При равенстве сохраняется первый признак и первый порог.
                    if costs[cut] < best_cost - 1e-12:
                        best_cost = float(costs[cut])
                        threshold = x[cut] / 2.0 + x[cut + 1] / 2.0
                        best = (feature, threshold, sorted_indices[:cut + 1], sorted_indices[cut + 1:])
            gain = 0.0 if best is None else max(0.0, parent_cost - best_cost)
            if best is None or gain / total_weight + 1e-15 < self.min_impurity_decrease:
                self.n_leaves_ += 1
                continue
            feature, threshold, left, right = best
            node.update(feature=feature, threshold=threshold)
            self.feature_importances_[feature] += gain
            pending.append((right, depth + 1, node_id, "right"))
            pending.append((left, depth + 1, node_id, "left"))
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        self.node_count_ = len(self.nodes_)
        return self

    def _values(self, X):
        check_is_fitted(self, "nodes_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        shape = (len(X), self.n_classes_) if self._classification else (len(X),)
        result = np.empty(shape)
        pending = [(0, np.arange(len(X)))]
        while pending:
            node_id, rows = pending.pop()
            if not len(rows):
                continue
            node = self.nodes_[node_id]
            if node["feature"] < 0:
                result[rows] = node["value"]
            else:
                goes_left = X[rows, node["feature"]] <= node["threshold"]
                pending.append((node["left"], rows[goes_left]))
                pending.append((node["right"], rows[~goes_left]))
        return result

    def get_depth(self):
        check_is_fitted(self, "nodes_")
        return self.max_depth_

    def get_n_leaves(self):
        check_is_fitted(self, "nodes_")
        return self.n_leaves_


class NumpyDecisionTreeClassifier(ClassifierMixin, _CART):
    """CART с Gini, вероятностями в листьях и необязательными весами классов."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, class_weight=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease)
        self.class_weight = class_weight

    def fit(self, X, y):
        return self._fit(X, y, classification=True)

    def predict_proba(self, X):
        return self._values(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class NumpyDecisionTreeRegressor(RegressorMixin, _CART):
    """CART с минимизацией SSE и средним целевой переменной в листе."""

    def fit(self, X, y):
        return self._fit(X, y, classification=False)

    def predict(self, X):
        return self._values(X)
