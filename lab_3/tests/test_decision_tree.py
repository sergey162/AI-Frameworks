"""Проверки критерия CART, весов, ограничений роста и API."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lab_3.src.decision_tree import NumpyDecisionTreeClassifier, NumpyDecisionTreeRegressor


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_weighted_stump_matches_exhaustive_search(weight):
    X = np.arange(12, dtype=float)[:, None]
    y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    w = np.ones(12) if weight is None else 12 / (2 * np.bincount(y)[y])
    costs = []
    for cut in range(2, 11):
        cost = 0
        for part in [slice(None, cut), slice(cut, None)]:
            counts = np.bincount(y[part], weights=w[part], minlength=2)
            cost += counts.sum() * (1 - np.sum((counts / counts.sum()) ** 2))
        costs.append(cost)
    expected = np.argmin(costs) + 2 - 0.5
    own = NumpyDecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight=weight).fit(X, y)
    assert own.nodes_[0]["threshold"] == expected
    ref = DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight=weight).fit(X, y)
    np.testing.assert_allclose(own.predict_proba(X), ref.predict_proba(X))


def test_regression_stump_sse_and_large_offset():
    X = np.arange(9)[:, None]
    y = np.array([1., 3., 2., 5., 10., 13., 11., 15., 12.]) + 1e9
    costs = [np.sum((y[:i] - y[:i].mean()) ** 2) + np.sum((y[i:] - y[i:].mean()) ** 2)
             for i in range(2, 8)]
    own = NumpyDecisionTreeRegressor(max_depth=1, min_samples_leaf=2).fit(X, y)
    assert own.nodes_[0]["threshold"] == np.argmin(costs) + 1.5
    for node in own.nodes_[1:]:
        assert node["n_samples"] >= 2


def test_regression_matches_sklearn_without_split_ties():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(140, 4))
    y = X[:, 0] ** 2 + 3 * X[:, 1] + rng.normal(size=140) * .05
    own = NumpyDecisionTreeRegressor(max_depth=4, min_samples_leaf=4).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=4, min_samples_leaf=4, random_state=42).fit(X, y)
    query = rng.normal(size=(50, 4))
    np.testing.assert_allclose(own.predict(query), ref.predict(query), atol=1e-10)
    np.testing.assert_allclose(own.feature_importances_, ref.feature_importances_, atol=1e-10)


def test_multiclass_labels_and_tied_votes():
    X = [[0], [1], [2], [3], [4], [5]]
    y = ["a", "b", "a", "b", "c", "c"]
    model = NumpyDecisionTreeClassifier(min_samples_split=10).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), np.full((6, 3), 1 / 3))
    assert set(model.predict(X)) == {"a"}
    assert model.get_depth() == 0 and model.get_n_leaves() == 1


def test_zero_gain_root_can_reveal_xor_interaction():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # Первый порог не уменьшает Gini, но второй уровень раскрывает XOR.
    model = NumpyDecisionTreeClassifier(max_depth=2).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.get_depth() == 2
    assert NumpyDecisionTreeClassifier(min_impurity_decrease=.01).fit(X, y).get_depth() == 0


@pytest.mark.parametrize("cls", [NumpyDecisionTreeClassifier, NumpyDecisionTreeRegressor])
def test_constant_features_validation_and_clone(cls):
    with pytest.raises(NotFittedError):
        cls().predict([[0]])
    model = clone(cls(max_depth=3, min_samples_leaf=2)).fit(np.ones((8, 2)), np.arange(8) % 2)
    assert model.get_depth() == 0
    assert model.node_count_ == 1
    np.testing.assert_array_equal(model.feature_importances_, [0, 0])
    with pytest.raises(ValueError):
        model.predict([[1, 2, 3]])
    for params in [dict(max_depth=0), dict(min_samples_leaf=0), dict(min_samples_split=1),
                   dict(min_impurity_decrease=-1), dict(max_depth=1.5)]:
        with pytest.raises(ValueError):
            cls(**params).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        cls().fit([[np.nan], [1]], [0, 1])


@pytest.mark.parametrize("cls", [NumpyDecisionTreeClassifier, NumpyDecisionTreeRegressor])
def test_growth_limits_and_no_improvement(cls):
    X = np.arange(80)[:, None]
    y = np.arange(80) % 3
    model = cls(max_depth=3, min_samples_leaf=5).fit(X, y)
    assert model.get_depth() <= 3
    assert all(n["n_samples"] >= 5 for n in model.nodes_ if n["feature"] < 0)
    stopped = cls(min_impurity_decrease=100).fit(X, y)
    assert stopped.node_count_ == 1
    single = cls().fit(X, np.ones(80))
    assert single.node_count_ == 1
