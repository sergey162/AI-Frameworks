"""Проверки математического поведения собственной реализации, включая вырожденные случаи."""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.knn import KNNClassifier, KNNRegressor


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_against_sklearn(p, weights):
    rng = np.random.default_rng(123)
    X, query = rng.normal(size=(73, 4)), rng.normal(size=(17, 4))
    labels = np.array(["a", "b", "c"])[rng.integers(0, 3, len(X))]
    target = rng.normal(size=len(X))
    params = dict(n_neighbors=7, weights=weights, p=p)
    own_c = KNNClassifier(**params, batch_size=5).fit(X, labels)
    ref_c = KNeighborsClassifier(**params, algorithm="brute").fit(X, labels)
    np.testing.assert_allclose(own_c.predict_proba(query), ref_c.predict_proba(query))
    np.testing.assert_array_equal(own_c.predict(query), ref_c.predict(query))
    own_r = KNNRegressor(**params, batch_size=5).fit(X, target)
    ref_r = KNeighborsRegressor(**params, algorithm="brute").fit(X, target)
    np.testing.assert_allclose(own_r.predict(query), ref_r.predict(query))


def test_zero_distance_and_class_tie():
    X = np.array([[0.], [0.], [2.]])
    reg = KNNRegressor(n_neighbors=3, weights="distance").fit(X, [2., 6., 100.])
    np.testing.assert_allclose(reg.predict([[0.]]), [4.])
    clf = KNNClassifier(n_neighbors=3, weights="distance").fit(X, [1, 0, 1])
    np.testing.assert_allclose(clf.predict_proba([[0.]]), [[.5, .5]])
    np.testing.assert_array_equal(clf.predict([[0.]]), [0])


def test_neighbor_boundary_tie_is_deterministic():
    model = KNNRegressor(n_neighbors=1).fit([[-1.], [1.]], [5., 9.])
    np.testing.assert_allclose(model.predict([[0.]]), [5.])


def test_validation_and_copy():
    with pytest.raises(NotFittedError):
        KNNRegressor().predict([[0.]])
    for params in [dict(n_neighbors=0), dict(n_neighbors=3), dict(p=3),
                   dict(weights="bad"), dict(batch_size=0)]:
        with pytest.raises(ValueError):
            KNNRegressor(**params).fit([[0.], [1.]], [0., 1.])
    X, y = np.array([[0.], [1.]]), np.array([2., 9.])
    model = KNNRegressor(n_neighbors=1).fit(X, y)
    X[:] = 100
    y[:] = 100
    np.testing.assert_allclose(model.predict([[0.]]), [2.])
    with pytest.raises(ValueError):
        model.predict([[0., 1.]])


def test_classifier_validation():
    # Ошибка до fit должна быть совместима с интерфейсом sklearn.
    for method in ["predict", "predict_proba"]:
        with pytest.raises(NotFittedError):
            getattr(KNNClassifier(), method)([[0.]])
    # Непрерывная цель относится к регрессии и не должна приниматься классификатором.
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=1).fit([[0.], [1.]], [0.2, 0.7])
