#!/usr/bin/env python
# coding: utf-8

# # Project - 190349K
# 
# ## Info
# 
# ### Datasets
# 
# - Set `DATA_C1` and `DATA_C2` to paths containing datasets
# - Set `MODEL_DIR` to directory where models should be saved
# 
# ### Saving and loading models
# 
# -   Models that are trained are also saved to `models/` in the `joblib` format
# -   Set `RETRAIN` to `False` to load saved models from `models/`
# 

# ## Loading and inspecting data

# In[ ]:


import gc
from enum import Enum
from typing import Dict

import numpy as np


class Label(Enum):
    """Labels of datasets"""

    C = "common"
    L1 = "label_1"
    L2 = "label_2"
    L3 = "label_3"
    L4 = "label_4"


class L(Enum):
    """Layers of the dataset"""

    C1 = "layer-7"
    C2 = "layer-12"


class K(Enum):
    """Kinds of datasets"""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


ID = "ID"
LABELS = [l.value for l in Label if l != Label.C]
AGE_LABEL = Label.L2
FEATURE_COUNT = 768
FEATURES = [f"feature_{i}" for i in range(1, FEATURE_COUNT + 1)]
RETRAIN = True  # Retrain the model or load the saved one
VERBOSE = True
RNG_SEED = 42
RNG = np.random.RandomState(RNG_SEED)

DATA_C1 = "data/layer-7"
DATA_C2 = "data/layer-12"
MODEL_DIR = "models"


# In[ ]:


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# In[ ]:


import pandas as pd

data: Dict[L, Dict[K, pd.DataFrame]] = {L.C1: {}, L.C2: {}}
data[L.C1][K.TRAIN] = pd.read_csv(f"{DATA_C1}/train.csv")
data[L.C1][K.VALID] = pd.read_csv(f"{DATA_C1}/valid.csv")
data[L.C1][K.TEST] = pd.read_csv(f"{DATA_C1}/test.csv")
data[L.C1][K.TRAIN].head()


# In[ ]:


data[L.C2][K.TRAIN] = pd.read_csv(f"{DATA_C2}/train.csv")
data[L.C2][K.VALID] = pd.read_csv(f"{DATA_C2}/valid.csv")
data[L.C2][K.TEST] = pd.read_csv(f"{DATA_C2}/test.csv")
data[L.C2][K.TRAIN].head()


# In[ ]:


data[L.C1][K.TRAIN][LABELS + FEATURES[::32]].describe()


# In[ ]:


data[L.C2][K.TRAIN][LABELS + FEATURES[::32]].describe()


# ## Preprocessing

# In[ ]:


LDfs = Dict[L, Dict[Label, pd.DataFrame]]
LSer = Dict[L, Dict[Label, pd.Series]]

# To store datasets
X_train: LDfs = {L.C1: {}, L.C2: {}}
X_valid: LDfs = {L.C1: {}, L.C2: {}}
X_test: LDfs = {L.C1: {}, L.C2: {}}
y_train: LSer = {L.C1: {}, L.C2: {}}
y_valid: LSer = {L.C1: {}, L.C2: {}}
y_pred_before: LSer = {L.C1: {}, L.C2: {}}
y_pred_after: LSer = {L.C1: {}, L.C2: {}}


def filter_missing_age(df: pd.DataFrame):
    """Filter out rows where age is `NaN`"""
    return df[df[AGE_LABEL.value].notna()]


# Separately store datasets
for layer in L:
    try:
        train_df = data[layer][K.TRAIN]
        valid_df = data[layer][K.VALID]
        test_df = data[layer][K.TEST]
    except:
        print(layer, "not found")

    X_train[layer][Label.C] = train_df.drop(LABELS, axis=1)
    X_valid[layer][Label.C] = valid_df.drop(LABELS, axis=1)
    X_test[layer][Label.C] = test_df.copy()

    for target_label in [Label.L1, Label.L2, Label.L3, Label.L4]:
        tr_df = filter_missing_age(train_df) if target_label == AGE_LABEL else train_df
        vl_df = filter_missing_age(valid_df) if target_label == AGE_LABEL else valid_df
        ts_df = test_df  # No need to filter rows with missing age in test dataset

        if target_label == AGE_LABEL:
            X_train[layer][target_label] = tr_df.drop(LABELS, axis=1)
            X_valid[layer][target_label] = vl_df.drop(LABELS, axis=1)
            X_test[layer][target_label] = ts_df.copy()
        else:
            # Only references to common dataframes
            X_train[layer][target_label] = X_train[layer][Label.C]
            X_valid[layer][target_label] = X_valid[layer][Label.C]
            X_test[layer][target_label] = X_test[layer][Label.C]

        y_train[layer][target_label] = tr_df[target_label.value]
        y_valid[layer][target_label] = vl_df[target_label.value]

del data
gc.collect()


# In[ ]:


X_train[L.C1][Label.L1].head()


# In[ ]:


y_train[L.C1][Label.L1].head()


# ## Model training

# In[ ]:


from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.base import BaseEstimator


# ### Predicting labels and showing statistics

# In[ ]:


from sklearn import metrics


def filter_nans(y_true: pd.Series, y_pred: pd.Series):
    """Filter `NaN`s in both `y_true` and `y_pred` based on `NaN`s in `y_true`"""
    return y_true[y_true.isna() == False], y_pred[y_true.isna() == False]


def predict(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred: pd.Series = model.predict(X_test)
    print("Stats:")
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
    print("F1:", metrics.f1_score(y_test, y_pred, average="weighted"))
    print(metrics.classification_report(y_test, y_pred))
    return y_pred


# ### Saving models

# In[ ]:


import joblib
import os


def save_model(model, name: str):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}/{name}.joblib", compress=True)


def load_model(name: str):
    return joblib.load(f"{MODEL_DIR}/{name}.joblib")


# ### Cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score


def cross_validate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv=5):
    log("Cross validating...")
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1, verbose=1)
    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (scores.mean(), scores.std())
    )
    return scores


# ### Hyperparameter tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

DEFAULT_SVC_PARAMS = {
    "C": [1, 10, 100, 1000],
    "gamma": ["scale", "auto", 1, 0.01, 0.0001],
}


def tune(
    base_estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict = DEFAULT_SVC_PARAMS,
    n_jobs=10,
    cv=5,
):
    """Tunes the hyperparameters of `base_estimator` using `GridSearchCV`"""
    log("Tuning...")
    verbosity = 4 if VERBOSE else 0
    sh = GridSearchCV(
        base_estimator,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbosity,
    ).fit(X, y)
    print(sh.best_params_)
    return sh.best_estimator_


# ## Training baseline models

# #### Competition 1 (layer 7)

# In[ ]:


# C1, L1

if RETRAIN:
    model = svm.SVC(kernel="rbf", random_state=RNG)
    model.fit(X_train[L.C1][Label.L1], y_train[L.C1][Label.L1])
    save_model(model, "c1_label_1_before")
else:
    model = load_model("c1_label_1_before")
predict(model, X_valid[L.C1][Label.L1], y_valid[L.C1][Label.L1])
y_pred_before[L.C1][Label.L1] = model.predict(X_test[L.C1][Label.L1].drop(ID, axis=1))


# In[ ]:


# C1, L2

if RETRAIN:
    model = svm.SVC(kernel="rbf", random_state=RNG)
    model.fit(X_train[L.C1][Label.L2], y_train[L.C1][Label.L2])
    save_model(model, "c1_label_2_before")
else:
    model = load_model("c1_label_2_before")
predict(model, X_valid[L.C1][Label.L2], y_valid[L.C1][Label.L2])
y_pred_before[L.C1][Label.L2] = model.predict(X_test[L.C1][Label.L2].drop(ID, axis=1))


# In[ ]:


# C1, L3

if RETRAIN:
    model = svm.SVC(kernel="rbf", random_state=RNG)
    model.fit(X_train[L.C1][Label.L3], y_train[L.C1][Label.L3])
    save_model(model, "c1_label_3_before")
else:
    model = load_model("c1_label_3_before")
predict(model, X_valid[L.C1][Label.L3], y_valid[L.C1][Label.L3])
y_pred_before[L.C1][Label.L3] = model.predict(X_test[L.C1][Label.L3].drop(ID, axis=1))


# In[ ]:


# C1, L4

if RETRAIN:
    model = svm.SVC(kernel="rbf", class_weight="balanced", random_state=RNG)
    model.fit(X_train[L.C1][Label.L4], y_train[L.C1][Label.L4])
    save_model(model, "c1_label_4_before")
else:
    model = load_model("c1_label_4_before")
predict(model, X_valid[L.C1][Label.L4], y_valid[L.C1][Label.L4])
y_pred_before[L.C1][Label.L4] = model.predict(X_test[L.C1][Label.L4].drop(ID, axis=1))


# #### Competition 2 (layer 12)

# In[ ]:


# C2, L1

if RETRAIN:
    model = svm.SVC(kernel="rbf")
    model.fit(X_train[L.C2][Label.L1], y_train[L.C2][Label.L1])
    save_model(model, "c2_label_1_before")
else:
    model = load_model("c2_label_1_before")
predict(model, X_valid[L.C2][Label.L1], y_valid[L.C2][Label.L1])
y_pred_before[L.C2][Label.L1] = model.predict(X_test[L.C2][Label.L1].drop(ID, axis=1))


# In[ ]:


# C2, L2

if RETRAIN:
    model = svm.SVC(kernel="rbf", C=1000)
    model.fit(X_train[L.C2][Label.L2], y_train[L.C2][Label.L2])
    save_model(model, "c2_label_2_before")
else:
    model = load_model("c2_label_2_before")
predict(model, X_valid[L.C2][Label.L2], y_valid[L.C2][Label.L2])
y_pred_before[L.C2][Label.L2] = model.predict(X_test[L.C2][Label.L2].drop(ID, axis=1))


# In[ ]:


# C2, L3

if RETRAIN:
    model = svm.SVC(kernel="rbf")
    model.fit(X_train[L.C2][Label.L3], y_train[L.C2][Label.L3])
    save_model(model, "c2_label_3_before")
else:
    model = load_model("c2_label_3_before")
predict(model, X_valid[L.C2][Label.L3], y_valid[L.C2][Label.L3])
y_pred_before[L.C2][Label.L3] = model.predict(X_test[L.C2][Label.L3].drop(ID, axis=1))


# In[ ]:


# C2, L4

if RETRAIN:
    model = svm.SVC(kernel="rbf", class_weight="balanced")
    model.fit(X_train[L.C2][Label.L4], y_train[L.C2][Label.L4])
    save_model(model, "c2_label_4_before")
else:
    model = load_model("c2_label_4_before")
predict(model, X_valid[L.C2][Label.L4], y_valid[L.C2][Label.L4])
y_pred_before[L.C2][Label.L4] = model.predict(X_test[L.C2][Label.L4].drop(ID, axis=1))


# ## With feature engineering

# ### Feature engineering functions

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def get_pca(pca_variance=0.95):
    return PCA(n_components=pca_variance, svd_solver="full", random_state=RNG)


def get_transformers(pca_count=5, pca_variance=0.95):
    return [(f"pca_{i}", get_pca(pca_variance)) for i in range(pca_count)]


# ### Competition 1 (layer 7)

# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    model = svm.SVC(kernel="rbf", C=100, gamma=0.0001, random_state=RNG)
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C1][Label.L1], y_train[L.C1][Label.L1])  # 40s
    save_model(pipeline, "c1_label_1_after")
else:
    pipeline = load_model("c1_label_1_after")
predict(pipeline, X_valid[L.C1][Label.L1], y_valid[L.C1][Label.L1])
y_pred_after[L.C1][Label.L1] = pipeline.predict(X_test[L.C1][Label.L1].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    model = svm.SVC(kernel="rbf", C=10, gamma="auto", random_state=RNG)
    pipeline = Pipeline(
        [
            # ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C1][Label.L2], y_train[L.C1][Label.L2])  # 1m 50s
    save_model(pipeline, "c1_label_2_after")
else:
    pipeline = load_model("c1_label_2_after")
predict(pipeline, X_valid[L.C1][Label.L2], y_valid[L.C1][Label.L2])
y_pred_after[L.C1][Label.L2] = pipeline.predict(X_test[L.C1][Label.L2].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    # model = svm.SVC(kernel="rbf", C=100, gamma='scale', random_state=RNG, verbose=True)
    model = svm.SVC(kernel="rbf", random_state=RNG)
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C1][Label.L3], y_train[L.C1][Label.L3])  # 40s
    save_model(pipeline, "c1_label_3_after")
else:
    pipeline = load_model("c1_label_3_after")
predict(pipeline, X_valid[L.C1][Label.L3], y_valid[L.C1][Label.L3])
y_pred_after[L.C1][Label.L3] = pipeline.predict(X_test[L.C1][Label.L3].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    # model = svm.SVC(kernel="rbf", class_weight="balanced", random_state=RNG)  # 5m
    model = svm.SVC(kernel="rbf", gamma="auto", random_state=RNG)  # 2m
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C1][Label.L4], y_train[L.C1][Label.L4])  # 2m
    save_model(pipeline, "c1_label_4_after")
else:
    pipeline = load_model("c1_label_4_after")
predict(pipeline, X_valid[L.C1][Label.L4], y_valid[L.C1][Label.L4])
y_pred_after[L.C1][Label.L4] = pipeline.predict(X_test[L.C1][Label.L4].drop(ID, axis=1))


# ### Competition 2 (layer 12)

# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    model = svm.SVC(kernel="rbf", C=1000, gamma="scale", random_state=RNG)  # 89%
    # model = svm.SVC(kernel="rbf", C=1000, gamma=0.0001, random_state=RNG)  # 86%
    # model = CatBoostClassifier(random_state=RNG_SEED)  # 86%
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C2][Label.L1], y_train[L.C2][Label.L1])  # 40s
    save_model(pipeline, "c2_label_1_after")
else:
    pipeline = load_model("c2_label_1_after")
predict(pipeline, X_valid[L.C2][Label.L1], y_valid[L.C2][Label.L1])
y_pred_after[L.C2][Label.L1] = pipeline.predict(X_test[L.C2][Label.L1].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    # model = svm.SVC(kernel="rbf", C=100, gamma='scale', random_state=RNG)
    model = CatBoostClassifier(random_state=RNG_SEED)  # 4m
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C2][Label.L2], y_train[L.C2][Label.L2])  # 4m
    save_model(pipeline, "c2_label_2_after")
else:
    pipeline = load_model("c2_label_2_after")
predict(pipeline, X_valid[L.C2][Label.L2], y_valid[L.C2][Label.L2])
y_pred_after[L.C2][Label.L2] = pipeline.predict(X_test[L.C2][Label.L2].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    # model = CatBoostClassifier(iterations=300, loss_function="MultiClass", max_depth=6, random_state=RNG_SEED)
    model = svm.SVC(kernel="rbf", gamma="scale", C=1, random_state=RNG)  # 20s
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C2][Label.L3], y_train[L.C2][Label.L3])  # 20s
    save_model(pipeline, "c2_label_3_after")
else:
    pipeline = load_model("c2_label_3_after")
predict(pipeline, X_valid[L.C2][Label.L3], y_valid[L.C2][Label.L3])
y_pred_after[L.C2][Label.L3] = pipeline.predict(X_test[L.C2][Label.L3].drop(ID, axis=1))


# In[ ]:


if RETRAIN:
    transformers = get_transformers(pca_count=1)
    model = svm.SVC(kernel="rbf", C=100, class_weight="balanced", random_state=RNG)
    # model = CatBoostClassifier(random_state=RNG_SEED)
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            *transformers,
            ("clf", model),
        ]
    )
    log("Training...")
    pipeline.fit(X_train[L.C2][Label.L4], y_train[L.C2][Label.L4])  # 1m
    save_model(pipeline, "c2_label_4_after")
else:
    pipeline = load_model("c2_label_4_after")
predict(pipeline, X_valid[L.C2][Label.L4], y_valid[L.C2][Label.L4])
y_pred_after[L.C2][Label.L4] = pipeline.predict(X_test[L.C2][Label.L4].drop(ID, axis=1))


# ## Results

# In[ ]:


print("X_test[layer-7][label_1]:", X_test[L.C1][Label.L1].shape)
print("y_pred[layer-7][label_1]:", y_pred_after[L.C1][Label.L1].shape)
print("X_test[layer-12][label_1]:", X_test[L.C2][Label.L1].shape)
print("y_pred[layer-12][label_1]:", y_pred_after[L.C2][Label.L1].shape)


# In[ ]:


result1 = pd.DataFrame(columns=[ID] + LABELS)
result1[ID] = X_test[L.C1][Label.L1][ID]
for label in Label:
    result1[label.value] = y_pred_after[L.C1][label].astype(int)


# In[ ]:


result2 = pd.DataFrame(columns=[ID] + LABELS)
result2[ID] = X_test[L.C2][Label.L1][ID]
for label in Label:
    result2[label.value] = y_pred_after[L.C2][label].astype(int)


# In[ ]:


result1.head()


# In[ ]:


result2.head()


# In[ ]:


result1.to_csv("results/layer-7.csv", index=False)


# In[ ]:


result2.to_csv("results/layer-12.csv", index=False)

