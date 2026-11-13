"""Общие разбиения №1–2 и подготовка данных для решающих деревьев."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lab_1.src.experiment import CATEGORICAL, SEED, TaskFeatures, evaluate
from lab_2.src.experiment import load_tasks


def sklearn_model(task, **params):
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(random_state=SEED, **params)


def make_pipeline(X, task, mode="raw", estimator=None):
    if mode not in ("raw", "scaled", "log", "engineered"):
        raise ValueError(f"Неизвестное представление {mode}")
    categorical = [column for column in CATEGORICAL if column in X]
    numeric = [column for column in X if column not in categorical]
    if mode == "engineered":
        numeric += (["employment_age_ratio", "credit_history_age_ratio"] if task == "classification"
                    else ["binder", "water_binder_ratio", "aggregate_binder_ratio"])
    steps = [("impute", SimpleImputer(strategy="median"))]
    if mode == "scaled":
        steps.append(("scale", StandardScaler()))
    transforms = [("numeric", Pipeline(steps), numeric)]
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ("features", TaskFeatures(task, mode)),
        ("preprocess", ColumnTransformer(transforms, sparse_threshold=0)),
        ("model", sklearn_model(task) if estimator is None else estimator),
    ])
