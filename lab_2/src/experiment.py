"""Общий протокол с лабораторной №1 и Pipeline линейных моделей."""

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from lab_1.src.experiment import (CATEGORICAL, DATASETS, SEED, TaskFeatures,
                                  evaluate, load_dataset, sha256)


def load_tasks(root):
    """Читаем сохранённые разбиения №1, проверяя SHA, полноту и группы."""
    lab = root / "lab_1"
    protocol = json.loads((lab / "results" / "protocol.json").read_text())
    tasks = {}
    for task, spec in DATASETS.items():
        raw = load_dataset(lab, task)
        data = raw.drop_duplicates()
        path = lab / "results" / f"split_{task}.csv"
        reference = protocol["datasets"][task]
        if sha256(path) != reference["split_sha"]:
            raise ValueError(f"Изменилось разбиение {task}")
        split = pd.read_csv(path)
        assert not split.row_id.duplicated().any()
        assert set(split.row_id) == set(data.index)
        assert set(split.split) == {"train", "test"}
        train = split.loc[split.split == "train", "row_id"].to_numpy()
        test = split.loc[split.split == "test", "row_id"].to_numpy()
        X, y = data.drop(columns=spec["excluded"]), data[spec["target"]]
        assert list(X.columns) == reference["features"]
        if task == "regression":
            groups = pd.Series(pd.factorize(pd.MultiIndex.from_frame(X.drop(columns="age")))[0], index=X.index)
        else:
            groups = pd.Series(X.index, index=X.index)
        np.testing.assert_array_equal(groups.loc[split.row_id], split.group_id)
        assert set(groups.loc[train]).isdisjoint(groups.loc[test])
        if task == "classification":
            folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X.loc[train], y.loc[train]))
        else:
            folds = list(GroupKFold(3).split(X.loc[train], y.loc[train], groups.loc[train]))
        for a, b in folds:
            assert set(groups.loc[train].iloc[a]).isdisjoint(groups.loc[train].iloc[b])
        tasks[task] = dict(X_train=X.loc[train], X_test=X.loc[test],
                           y_train=y.loc[train], y_test=y.loc[test], cv=folds,
                           split=split, spec=spec, split_sha=sha256(path),
                           duplicates_removed=len(raw) - len(data))
    return tasks


def sklearn_model(task, **params):
    if task == "classification":
        return LogisticRegression(solver="newton-cholesky", max_iter=100, tol=1e-9, **params)
    alpha = params.pop("alpha", 0.0)
    return LinearRegression(**params) if alpha == 0 else Ridge(alpha=alpha, solver="svd", **params)


def make_pipeline(X, task, mode="scaled", estimator=None):
    if mode not in ("scaled", "log", "engineered", "polynomial"):
        raise ValueError(f"Неизвестное представление {mode}")
    categorical = [column for column in CATEGORICAL if column in X]
    numeric = [column for column in X if column not in categorical]
    feature_mode = "engineered" if mode == "polynomial" else mode
    if feature_mode == "engineered":
        numeric += (["employment_age_ratio", "credit_history_age_ratio"] if task == "classification"
                    else ["binder", "water_binder_ratio", "aggregate_binder_ratio"])
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if mode == "polynomial":
        steps += [("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                  ("rescale", StandardScaler())]
    transforms = [("numeric", Pipeline(steps), numeric)]
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ("features", TaskFeatures(task, feature_mode)),
        ("preprocess", ColumnTransformer(transforms, sparse_threshold=0)),
        ("model", sklearn_model(task) if estimator is None else estimator),
    ])
