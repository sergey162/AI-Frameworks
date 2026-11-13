"""Градиентный бустинг на NumPy поверх собственной CART из лабораторной №3.

Готовые деревья и ансамбли sklearn не используются. Бинарная log-loss и
квадратичная ошибка, shrinkage, subsampling без возвращения, staged-прогнозы.
"""

from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data
from lab_3.src.decision_tree import NumpyDecisionTreeRegressor


def sigmoid(raw):
    """Устойчивая sigmoid: exp получает только неположительные аргументы."""
    raw = np.asarray(raw, dtype=float)
    z = np.exp(-np.abs(raw))
    return np.where(raw >= 0, 1 / (1 + z), z / (1 + z))


class _BoostingTree(NumpyDecisionTreeRegressor):
    def apply(self, X):
        """Идентификаторы листьев для обновления Newton; обход собственного CART."""
        check_is_fitted(self, "nodes_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        leaves = np.empty(len(X), dtype=int)
        pending = [(0, np.arange(len(X)))]
        while pending:
            node_id, rows = pending.pop()
            if not len(rows):
                continue
            node = self.nodes_[node_id]
            if node["feature"] < 0:
                leaves[rows] = node_id
            else:
                left = X[rows, node["feature"]] <= node["threshold"]
                pending.append((node["left"], rows[left]))
                pending.append((node["right"], rows[~left]))
        return leaves


class _GradientBoosting(BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def _validate_parameters(self):
        for name in ["n_estimators", "max_depth", "min_samples_leaf"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} должен быть целым >= 1")
        for name in ["learning_rate", "subsample"]:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} должен быть конечным положительным числом")
        if self.subsample > 1:
            raise ValueError("subsample должен принадлежать (0, 1]")
        if self.random_state is not None and (isinstance(self.random_state, bool)
                or not isinstance(self.random_state, Integral) or self.random_state < 0):
            raise ValueError("random_state должен быть целым >= 0 или None")

    def _fit(self, X, y, classification):
        self._validate_parameters()
        X, y = validate_data(self, X, y, dtype=np.float32, y_numeric=not classification)
        if y.ndim != 1:
            raise ValueError("Поддерживается одна целевая переменная")
        self._classification = classification
        if classification:
            check_classification_targets(y)
            self.classes_, encoded = np.unique(y, return_inverse=True)
            if len(self.classes_) != 2:
                raise ValueError("Поддерживается только бинарная классификация с двумя классами")
            target = encoded.astype(float)
            prior = np.clip(target.mean(), np.finfo(float).eps, 1 - np.finfo(float).eps)
            self.init_ = float(np.log(prior / (1 - prior)))
        else:
            target = np.asarray(y, dtype=float)
            self.init_ = float(target.mean())
        raw = np.full(len(target), self.init_)
        rng = np.random.default_rng(self.random_state)
        self.estimators_, self.estimators_samples_, self.train_score_ = [], [], []
        self.n_samples_subsample_ = max(1, int(self.subsample * len(target)))
        for _ in range(self.n_estimators):
            # Остатки считаются по ансамблю предыдущего шага, до добавления дерева.
            probability = sigmoid(raw) if classification else None
            gradient = target - (probability if classification else raw)
            rows = (rng.choice(len(target), self.n_samples_subsample_, replace=False)
                    if self.subsample < 1 else np.arange(len(target)))
            tree = _BoostingTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[rows], gradient[rows])
            if classification:
                # Дерево строит регионы по SSE градиента; значения листьев заменяет Newton.
                leaves = tree.apply(X[rows])
                for leaf in np.unique(leaves):
                    indices = rows[leaves == leaf]
                    numerator = gradient[indices].sum()
                    denominator = np.sum(probability[indices] * (1 - probability[indices]))
                    tree.nodes_[int(leaf)]["value"] = (float(numerator / denominator)
                        if denominator > np.finfo(float).eps else 0.0)
            raw += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.estimators_samples_.append(rows)
            loss = (np.mean(np.logaddexp(0, raw) - target * raw) if classification
                    else np.mean((target - raw) ** 2))
            self.train_score_.append(float(loss))
        self.train_score_ = np.asarray(self.train_score_)
        self.n_estimators_ = len(self.estimators_)
        return self

    def _staged_raw_predict(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        raw = np.full(len(X), self.init_)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
            yield raw.copy()

    def _raw_predict(self, X):
        # Последний элемент staged; копии защищают историю от изменения in-place.
        for raw in self._staged_raw_predict(X):
            pass
        return raw


class NumpyGradientBoostingClassifier(ClassifierMixin, _GradientBoosting):
    """Бинарная log-loss, градиент y-p и Newton-обновление каждого листа."""

    def fit(self, X, y):
        return self._fit(X, y, classification=True)

    def decision_function(self, X):
        return self._raw_predict(X)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def staged_predict_proba(self, X):
        for raw in self._staged_raw_predict(X):
            p = sigmoid(raw)
            yield np.column_stack([1 - p, p])

    def staged_predict(self, X):
        for raw in self._staged_raw_predict(X):
            yield self.classes_[(raw > 0).astype(int)]


class NumpyGradientBoostingRegressor(RegressorMixin, _GradientBoosting):
    """Квадратичная ошибка: начальное среднее и деревья остатков y-F."""

    def fit(self, X, y):
        return self._fit(X, y, classification=False)

    def predict(self, X):
        return self._raw_predict(X)

    def staged_predict(self, X):
        yield from self._staged_raw_predict(X)
