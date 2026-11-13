"""Случайный лес на NumPy: bootstrap, случайные признаки в узлах и OOB.

Критерии, поиск порогов и рост CART перенесены из собственной реализации №3.
Готовые деревья и ансамбли sklearn не используются; sklearn нужен для API/валидации.
Bootstrap представлен повторяющимися строками: min_samples_* считают их с повторами.
"""

from numbers import Integral, Real
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted, validate_data


class _RandomizedTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, classification, weights, n_classes=None):
        rng = np.random.default_rng(self.random_state)
        self._classification = classification
        if classification:
            target = y
            self.n_classes_ = n_classes
        else:
            # Центрирование уменьшает потерю точности при вычислении SSE.
            self.target_offset_ = float(np.mean(y))
            target = np.asarray(y, dtype=np.float64) - self.target_offset_
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
                # Каждый узел получает новый случайный набор признаков.
                # Если первые m признаков не допускают порога, пробуем следующие.
                for attempt, feature in enumerate(rng.permutation(X.shape[1])):
                    if attempt >= self.max_features and best is not None:
                        break
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


class _Forest(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, bootstrap=True,
                 max_samples=None, oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.random_state = random_state

    def _validate_parameters(self, n_samples, n_features):
        for name, lower in [("n_estimators", 1), ("max_depth", 1),
                            ("min_samples_split", 2), ("min_samples_leaf", 1)]:
            value = getattr(self, name)
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
                raise ValueError(f"{name} должен быть целым >= {lower}")
        if not isinstance(self.bootstrap, bool) or not isinstance(self.oob_score, bool):
            raise ValueError("bootstrap и oob_score должны быть bool")
        if not self.bootstrap and (self.oob_score or self.max_samples is not None):
            raise ValueError("OOB и max_samples требуют bootstrap=True")
        if self.random_state is not None and (isinstance(self.random_state, bool)
                or not isinstance(self.random_state, Integral) or self.random_state < 0):
            raise ValueError("random_state должен быть неотрицательным целым или None")
        m = self.max_features
        if m is None:
            self.max_features_ = n_features
        elif isinstance(m, str) and m in ("sqrt", "log2"):
            self.max_features_ = max(1, int(np.sqrt(n_features) if m == "sqrt" else np.log2(n_features)))
        elif not isinstance(m, bool) and isinstance(m, Integral) and 1 <= m <= n_features:
            self.max_features_ = int(m)
        elif not isinstance(m, bool) and isinstance(m, Real) and 0 < m <= 1:
            self.max_features_ = max(1, int(m * n_features))
        else:
            raise ValueError("Некорректное max_features")
        m = self.max_samples
        if m is None:
            self.n_samples_bootstrap_ = n_samples
        elif not isinstance(m, bool) and isinstance(m, Integral) and 1 <= m <= n_samples:
            self.n_samples_bootstrap_ = int(m)
        elif not isinstance(m, bool) and isinstance(m, Real) and 0 < m <= 1:
            self.n_samples_bootstrap_ = max(1, round(m * n_samples))
        else:
            raise ValueError("Некорректное max_samples")

    def _fit(self, X, y, classification):
        X, y = validate_data(self, X, y, dtype=np.float32, y_numeric=not classification)
        if y.ndim != 1:
            raise ValueError("Поддерживается одна целевая переменная")
        self._validate_parameters(len(X), X.shape[1])
        self._classification = classification
        if classification:
            check_classification_targets(y)
            if self.class_weight not in (None, "balanced", "balanced_subsample"):
                raise ValueError("class_weight: None, balanced или balanced_subsample")
            self.classes_, target = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            global_counts = np.bincount(target, minlength=self.n_classes_)
        else:
            target = np.asarray(y, dtype=float)
        # Удаляем старые OOB-атрибуты при повторном fit с oob_score=False.
        for name in ["oob_score_", "oob_prediction_", "oob_decision_function_",
                     "oob_counts_", "oob_coverage_"]:
            self.__dict__.pop(name, None)
        rng = np.random.default_rng(self.random_state)
        self.estimators_, self.estimators_samples_, self.tree_seeds_ = [], [], []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            sample_rng = np.random.default_rng(seed)
            indices = (sample_rng.integers(0, len(X), size=self.n_samples_bootstrap_)
                       if self.bootstrap else np.arange(len(X)))
            weights = np.ones(len(indices))
            if classification and self.class_weight is not None:
                counts = (np.bincount(target[indices], minlength=self.n_classes_)
                          if self.class_weight == "balanced_subsample" else global_counts)
                total = len(indices) if self.class_weight == "balanced_subsample" else len(X)
                # Отсутствующий в bootstrap класс не даёт деления на ноль.
                weights = total / (self.n_classes_ * counts[target[indices]])
            tree = _RandomizedTree(self.max_depth, self.min_samples_split,
                                   self.min_samples_leaf, max_features=self.max_features_,
                                   random_state=seed)
            tree.fit(X[indices], target[indices], classification, weights,
                     self.n_classes_ if classification else None)
            self.estimators_.append(tree)
            self.estimators_samples_.append(indices)
            self.tree_seeds_.append(seed)
        importance = np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0)
        self.feature_importances_ = importance / importance.sum() if importance.sum() else importance
        if self.oob_score:
            self._compute_oob(X, target)
        return self

    def _compute_oob(self, X, y):
        shape = (len(X), self.n_classes_) if self._classification else (len(X),)
        sums = np.zeros(shape)
        self.oob_counts_ = np.zeros(len(X), dtype=int)
        for tree, samples in zip(self.estimators_, self.estimators_samples_):
            rows = np.flatnonzero(np.bincount(samples, minlength=len(X)) == 0)
            if len(rows):
                sums[rows] += tree._values(X[rows])
                self.oob_counts_[rows] += 1
        valid = self.oob_counts_ > 0
        self.oob_coverage_ = float(valid.mean())
        result = np.full(shape, np.nan)
        divisor = self.oob_counts_[valid, None] if self._classification else self.oob_counts_[valid]
        result[valid] = sums[valid] / divisor
        if not valid.all():
            warnings.warn("Некоторые строки не имеют OOB-деревьев: прогноз для них NaN", UserWarning)
        if self._classification:
            self.oob_decision_function_ = result
            self.oob_score_ = accuracy_score(y[valid], result[valid].argmax(axis=1)) if valid.any() else np.nan
        else:
            self.oob_prediction_ = result
            self.oob_score_ = r2_score(y[valid], result[valid]) if valid.sum() >= 2 else np.nan

    def _predict_values(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        # Суммирование без хранения тензора n_trees × n_samples × n_classes.
        result = self.estimators_[0]._values(X).copy()
        for tree in self.estimators_[1:]:
            result += tree._values(X)
        return result / len(self.estimators_)


class NumpyRandomForestClassifier(ClassifierMixin, _Forest):
    """Лес Gini; усредняются вероятности листьев, при равенстве — первый класс."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 max_samples=None, oob_score=False, random_state=None, class_weight=None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, max_samples, oob_score, random_state)
        self.class_weight = class_weight

    def fit(self, X, y):
        return self._fit(X, y, classification=True)

    def predict_proba(self, X):
        return self._predict_values(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class NumpyRandomForestRegressor(RegressorMixin, _Forest):
    """Лес squared_error; ответ — среднее прогнозов CART."""

    def fit(self, X, y):
        return self._fit(X, y, classification=False)

    def predict(self, X):
        return self._predict_values(X)
