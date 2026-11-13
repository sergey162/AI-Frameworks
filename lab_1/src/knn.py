"""KNN на NumPy: собственные расстояния, поиск соседей и агрегация ответов."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform", p=2, batch_size=128):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.batch_size = batch_size

    def _fit(self, X, y):
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError("n_neighbors должен быть положительным целым")
        if self.weights not in ("uniform", "distance") or self.p not in (1, 2):
            raise ValueError("Поддерживаются uniform/distance и p=1/2")
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError("batch_size должен быть положительным целым")
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_2d=True)
        if y.ndim != 1:
            raise ValueError("Поддерживается одна целевая переменная")
        if self.n_neighbors > len(X):
            raise ValueError("Число соседей превышает размер обучающей выборки")
        # Копии защищают обученную модель от изменения исходных массивов.
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def _neighbors(self, X):
        check_is_fitted(self, ["X_", "y_"])
        X = validate_data(self, X, reset=False, dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            block = X[start:start + self.batch_size]
            # Накапливаем расстояние по признакам; не создаём тензор (B, N, D).
            distances = np.zeros((len(block), len(self.X_)), dtype=np.float64)
            for column in range(self.n_features_in_):
                delta = block[:, column, None] - self.X_[None, :, column]
                distances += np.abs(delta) if self.p == 1 else delta * delta
            # При совпадении расстояний на границе k выбираем ранний индекс train.
            # Быстрый argpartition дополнен явным разрешением граничных ничьих.
            indices = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, :self.n_neighbors]
            for row in range(len(block)):
                boundary = distances[row, indices[row]].max()
                closer = np.flatnonzero(distances[row] < boundary)
                equal = np.flatnonzero(distances[row] == boundary)
                indices[row] = np.concatenate((closer, equal[:self.n_neighbors - len(closer)]))
            selected = np.take_along_axis(distances, indices, axis=1)
            if self.p == 2:
                selected = np.sqrt(selected)
            if self.weights == "uniform":
                weights = np.ones_like(selected)
            else:
                zero = selected == 0
                # Если есть точные совпадения, учитываем только их, без деления на 0.
                weights = np.divide(1.0, selected, out=np.zeros_like(selected), where=~zero)
                has_zero = zero.any(axis=1)
                weights[has_zero] = zero[has_zero]
            yield indices, weights


class KNNClassifier(ClassifierMixin, _KNNBase):
    def fit(self, X, y):
        self._fit(X, y)
        check_classification_targets(self.y_)
        self.classes_, self.encoded_y_ = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba(self, X):
        blocks = []
        for indices, weights in self._neighbors(X):
            labels = self.encoded_y_[indices]
            votes = np.column_stack([
                (weights * (labels == label)).sum(axis=1)
                for label in range(len(self.classes_))
            ])
            blocks.append(votes / weights.sum(axis=1, keepdims=True))
        return np.vstack(blocks)

    def predict(self, X):
        # np.argmax разрешает равенство голосов в пользу первого класса np.unique.
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class KNNRegressor(RegressorMixin, _KNNBase):
    def fit(self, X, y):
        self._fit(X, y)
        self.y_ = np.asarray(self.y_, dtype=np.float64)
        if not np.isfinite(self.y_).all():
            raise ValueError("y должен содержать конечные числа")
        return self

    def predict(self, X):
        return np.concatenate([
            (self.y_[indices] * weights).sum(axis=1) / weights.sum(axis=1)
            for indices, weights in self._neighbors(X)
        ])
