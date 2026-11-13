"""Проверки bootstrap, OOB без in-bag строк, агрегации и рандомизации CART."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor
from lab_4.src.random_forest import NumpyRandomForestClassifier, NumpyRandomForestRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(90, 4))
    y = X[:, 0] ** 2 + 2 * X[:, 1] + rng.normal(size=90) * .01
    return X, y


@pytest.mark.parametrize("cls", [NumpyRandomForestClassifier, NumpyRandomForestRegressor])
def test_seed_bootstrap_and_refit(cls, data):
    X, y = data
    if cls is NumpyRandomForestClassifier:
        y = np.where(y > np.median(y), "yes", "no")
    first = cls(n_estimators=8, max_depth=3, random_state=42).fit(X, y)
    second = clone(first).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    for a, b in zip(first.estimators_samples_, second.estimators_samples_):
        np.testing.assert_array_equal(a, b)
        assert len(a) == len(X) and len(np.unique(a)) < len(X)
    assert not np.array_equal(first.estimators_samples_[0], first.estimators_samples_[1])
    first.fit(X, y)
    assert len(first.estimators_) == 8  # fit не накапливает предыдущие деревья.
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_regression_mean_and_single_tree_reference(data):
    X, y = data
    model = NumpyRandomForestRegressor(n_estimators=4, bootstrap=False,
        max_depth=3, min_samples_leaf=3, max_features=None, random_state=5).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=3, min_samples_leaf=3).fit(X, y)
    np.testing.assert_allclose(model.predict(X), ref.predict(X), atol=1e-10)
    np.testing.assert_allclose(model.feature_importances_, ref.feature_importances_, atol=1e-10)


@pytest.mark.parametrize("classification", [True, False])
def test_oob_is_independent_reconstruction_from_unsampled_rows(data, classification):
    X, y = data
    cls = NumpyRandomForestClassifier if classification else NumpyRandomForestRegressor
    if classification:
        y = (y > np.median(y)).astype(int)
    model = cls(n_estimators=25, max_depth=3, max_samples=.7,
                oob_score=True, random_state=17).fit(X, y)
    sums = np.zeros((len(X), 2)) if classification else np.zeros(len(X))
    counts = np.zeros(len(X), dtype=int)
    for row in range(len(X)):
        for tree, samples in zip(model.estimators_, model.estimators_samples_):
            if row not in samples:
                sums[row] += tree._values(X[row:row + 1].astype(np.float32))[0]
                counts[row] += 1
    expected = sums / (counts[:, None] if classification else counts)
    actual = model.oob_decision_function_ if classification else model.oob_prediction_
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(model.oob_counts_, counts)
    assert model.oob_coverage_ == 1


def test_probabilities_average_with_missing_bootstrap_classes():
    X = np.arange(18)[:, None]
    y = np.array(["a"] * 16 + ["b", "c"])
    model = NumpyRandomForestClassifier(n_estimators=20, max_samples=2,
        class_weight="balanced_subsample", random_state=8).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (18, 3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert np.all(p >= 0)
    # Проверка математической агрегации мягких, а не жёстких голосов.
    expected = np.mean([t._values(X.astype(np.float32)) for t in model.estimators_], axis=0)
    np.testing.assert_allclose(p, expected)
    np.testing.assert_array_equal(model.predict(X), model.classes_[expected.argmax(axis=1)])


@pytest.mark.parametrize("weight", ["balanced", "balanced_subsample"])
def test_class_weights_in_unsplit_leaf(weight):
    X = np.ones((20, 2))
    y = [0] * 16 + [1] * 4
    model = NumpyRandomForestClassifier(n_estimators=3, bootstrap=False,
        class_weight=weight, random_state=12).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), .5)
    assert np.all(model.predict(X) == 0)
    np.testing.assert_array_equal(model.feature_importances_, 0)


def test_random_features_fallback_and_limits():
    # Константные признаки не должны мешать найти допустимый порог.
    X = np.zeros((60, 10))
    X[:, 7] = np.arange(60)
    y = (X[:, 7] >= 30).astype(int)
    model = NumpyRandomForestClassifier(n_estimators=3, max_features=1, bootstrap=False,
        max_depth=2, min_samples_leaf=5, random_state=9).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    for tree in model.estimators_:
        assert tree.get_depth() <= 2
        assert all(n["n_samples"] >= 5 for n in tree.nodes_ if n["feature"] < 0)


def test_feature_subset_changes_between_nodes():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = NumpyRandomForestClassifier(n_estimators=1, max_features=1,
        bootstrap=False, max_depth=2, random_state=4).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert {n["feature"] for n in model.estimators_[0].nodes_ if n["feature"] >= 0} == {0, 1}


def test_uncovered_oob_rows_and_clearing_on_refit():
    model = NumpyRandomForestRegressor(n_estimators=1, oob_score=True, random_state=1)
    with pytest.warns(UserWarning, match="OOB"):
        model.fit([[0], [1]], [1., 2.])
    assert np.isnan(model.oob_prediction_[model.oob_counts_ == 0]).all()
    model.set_params(oob_score=False).fit([[0], [1]], [1., 2.])
    assert not hasattr(model, "oob_prediction_")


@pytest.mark.parametrize("cls", [NumpyRandomForestClassifier, NumpyRandomForestRegressor])
def test_api_and_invalid_parameters(cls):
    with pytest.raises(NotFittedError):
        cls().predict([[0]])
    invalid = [dict(n_estimators=0), dict(max_depth=0), dict(min_samples_leaf=0),
        dict(min_samples_split=1), dict(max_features=0), dict(max_features=3),
        dict(max_features="bad"), dict(max_samples=0), dict(max_samples=1.1),
        dict(bootstrap=False, oob_score=True), dict(bootstrap=False, max_samples=.5),
        dict(random_state=-1), dict(n_estimators=True)]
    for params in invalid:
        with pytest.raises(ValueError):
            cls(**params).fit([[0, 0], [1, 1]], [0, 1])
    with pytest.raises(ValueError):
        cls(n_estimators=1).fit([[np.nan], [1]], [0, 1])
