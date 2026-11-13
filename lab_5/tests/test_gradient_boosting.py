"""Математические проверки бустинга, включая ручной Newton и перенос CART."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from lab_5.src.gradient_boosting import (NumpyGradientBoostingClassifier as Classifier,
    NumpyGradientBoostingRegressor as Regressor, sigmoid)


def test_one_regression_step_matches_manual_residuals():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([1., 1., 5., 5.])
    m = Regressor(n_estimators=1, max_depth=1, learning_rate=.25).fit(X, y)
    assert m.init_ == 3
    np.testing.assert_allclose(m.predict(X), [2.5, 2.5, 3.5, 3.5])


def test_one_logistic_step_matches_manual_newton():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    m = Classifier(n_estimators=1, max_depth=1, learning_rate=.1).fit(X, y)
    np.testing.assert_allclose(m.decision_function(X), [-.2, -.2, .2, .2])
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], sigmoid(np.array([-.2, -.2, .2, .2])))


@pytest.mark.parametrize("cls,reference", [(Classifier, GradientBoostingClassifier),
                                            (Regressor, GradientBoostingRegressor)])
def test_multiple_steps_match_sklearn_on_unambiguous_splits(cls, reference):
    # Один признак, нет неоднозначности выбора между дублирующими признаками.
    rng = np.random.default_rng(91)
    X = rng.normal(size=(97, 1))
    y = ((X[:, 0] + rng.normal(size=97) * .7) > .2).astype(int) if cls is Classifier else 2 * X[:, 0] ** 2 + rng.normal(size=97)
    params = dict(n_estimators=8, learning_rate=.13, max_depth=2, min_samples_leaf=3, random_state=42)
    custom = cls(**params).fit(X, y)
    ref = reference(**params).fit(X, y)
    method = "predict_proba" if cls is Classifier else "predict"
    np.testing.assert_allclose(getattr(custom, method)(X), getattr(ref, method)(X), atol=1e-10)


@pytest.mark.parametrize("cls", [Classifier, Regressor])
def test_stages_subsampling_repeat_fit_and_clone(cls):
    X = np.arange(60).reshape(30, 2)
    y = np.tile([0, 1, 0], 10) if cls is Classifier else np.sin(np.arange(30))
    m = cls(n_estimators=4, subsample=.7, random_state=7).fit(X, y)
    first = m.predict(X)
    samples = [s.copy() for s in m.estimators_samples_]
    assert all(len(s) == len(set(s)) == 21 for s in samples)
    stages = list(m.staged_predict(X))
    np.testing.assert_allclose(stages[-1], first)
    assert not np.shares_memory(stages[0], stages[-1])
    m.fit(X, y)
    assert len(m.estimators_) == 4
    np.testing.assert_allclose(m.predict(X), first)
    for a, b in zip(samples, m.estimators_samples_):
        np.testing.assert_array_equal(a, b)
    assert clone(m).get_params() == m.get_params()


def test_labels_probabilities_tie_and_numerical_stability():
    X = np.ones((4, 2))
    m = Classifier(n_estimators=3).fit(X, ["yes", "no", "yes", "no"])
    assert np.all(m.predict(X) == "no")
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1)
    np.testing.assert_allclose(sigmoid(np.array([-1e9, 0, 1e9])), [0, .5, 1])
    assert np.isfinite(m.train_score_).all()
    with pytest.raises(ValueError, match="бинарная"):
        Classifier().fit(X, [0, 1, 2, 0])
    with pytest.raises(ValueError, match="бинарная"):
        Classifier().fit(X, [1, 1, 1, 1])


def test_constant_regression_and_shift():
    X = np.arange(20).reshape(-1, 1)
    y = np.sin(np.arange(20))
    a = Regressor(n_estimators=7).fit(X, y)
    b = Regressor(n_estimators=7).fit(X, y + 1e6)
    np.testing.assert_allclose(a.predict(X) + 1e6, b.predict(X), atol=1e-8)
    np.testing.assert_allclose(Regressor(n_estimators=3).fit(X, np.full(20, 9.)).predict(X), 9)


@pytest.mark.parametrize("params", [{"n_estimators": 0}, {"max_depth": True},
    {"min_samples_leaf": 0}, {"learning_rate": np.nan}, {"subsample": 1.1},
    {"subsample": 0}, {"random_state": -1}])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        Regressor(**params).fit([[0], [1]], [0, 1])


def test_not_fitted_and_wrong_feature_count():
    with pytest.raises(NotFittedError):
        Regressor().predict([[0]])
    model = Regressor(n_estimators=1).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([[0, 1]])
