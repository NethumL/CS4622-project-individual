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

# In[20]:


from enum import Enum
from typing import Dict

import numpy as np


class Label(Enum):
    """Labels of datasets"""

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
LABELS = [l.value for l in Label]
AGE_LABEL = Label.L2
FEATURE_COUNT = 768
FEATURES = [f"feature_{i}" for i in range(1, FEATURE_COUNT + 1)]
RETRAIN = True  # Retrain the model or load the saved one
VERBOSE = True
RNG_SEED = 42

DATA_C1 = "data/layer-7"
DATA_C2 = "data/layer-12"
MODEL_DIR = "models"

RNG = np.random.RandomState(RNG_SEED)


# In[21]:


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# In[22]:


import pandas as pd

data: Dict[L, Dict[K, pd.DataFrame]] = {L.C1: {}, L.C2: {}}
data[L.C1][K.TRAIN] = pd.read_csv(f"{DATA_C1}/train.csv")
data[L.C1][K.VALID] = pd.read_csv(f"{DATA_C1}/valid.csv")
data[L.C1][K.TEST] = pd.read_csv(f"{DATA_C1}/test.csv")
data[L.C1][K.TRAIN].head()


# In[23]:


data[L.C2][K.TRAIN] = pd.read_csv(f"{DATA_C2}/train.csv")
data[L.C2][K.VALID] = pd.read_csv(f"{DATA_C2}/valid.csv")
data[L.C2][K.TEST] = pd.read_csv(f"{DATA_C2}/test.csv")
data[L.C2][K.TRAIN].head()


# In[24]:


data[L.C2][K.TRAIN][LABELS + FEATURES[::32]].describe()


# In[25]:


data[L.C2][K.VALID][LABELS + FEATURES[::32]].describe()


# ## Preprocessing

# In[26]:


from sklearn.preprocessing import RobustScaler

LDfs = Dict[L, Dict[Label, pd.DataFrame]]
LSer = Dict[L, Dict[Label, pd.Series]]

# To store datasets for each label
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


# Filter `NaN` and scale datasets
for layer in [L.C1, L.C2]:
    try:
        train_df = data[layer][K.TRAIN]
        valid_df = data[layer][K.VALID]
        test_df = data[layer][K.TEST]
    except:
        print(layer, "not found")
    for target_label in Label:
        tr_df = filter_missing_age(train_df) if target_label == AGE_LABEL else train_df
        vl_df = filter_missing_age(valid_df) if target_label == AGE_LABEL else valid_df
        ts_df = test_df  # No need to filter rows with missing age in test dataset

        scaler = RobustScaler()
        scaler.fit(tr_df.drop(LABELS, axis=1))
        X_train[layer][target_label] = pd.DataFrame(
            scaler.transform(tr_df.drop(LABELS, axis=1)), columns=FEATURES
        )
        y_train[layer][target_label] = tr_df[target_label.value]
        X_valid[layer][target_label] = pd.DataFrame(
            scaler.transform(vl_df.drop(LABELS, axis=1)), columns=FEATURES
        )
        y_valid[layer][target_label] = vl_df[target_label.value]
        X_test[layer][target_label] = pd.DataFrame(
            scaler.transform(ts_df.drop(ID, axis=1)), columns=FEATURES
        )
        X_test[layer][target_label][ID] = ts_df[ID]
del data


# In[27]:


X_train[L.C1][Label.L1].head()


# In[28]:


y_train[L.C1][Label.L1].head()


# ## Model training

# In[29]:


from sklearn import svm
from catboost import CatBoostClassifier


# ### Predicting labels and showing statistics

# In[30]:


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

# In[31]:


import joblib
import os


def save_model(model, name: str):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}/{name}.joblib", compress=True)


def load_model(name: str):
    return joblib.load(f"{MODEL_DIR}/{name}.joblib")


# ### Cross validation

# In[32]:


from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator


def cross_validate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series):
    log("Cross validating...")
    scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (scores.mean(), scores.std())
    )
    return scores


# ### Hyperparameter tuning

# In[33]:


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def tune(base_estimator, X, y, param_grid: dict = {}, n_jobs=10):
    """Tunes the hyperparameters of `base_estimator` using `HalvingGridSearchCV`"""
    log("Tuning...")
    if len(param_grid) == 0:
        param_grid = {
            "C": [1, 10, 100, 1000],
            "gamma": ["scale", "auto", 1, 0.01, 0.0001],
        }
    verbosity = 2 if VERBOSE else 0
    sh = HalvingGridSearchCV(
        base_estimator,
        param_grid,
        cv=5,
        factor=2,
        n_jobs=n_jobs,
        verbose=verbosity,
        random_state=RNG,
    ).fit(X, y)
    print(sh.best_params_)


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
    model.fit(X_train[L.C1][Label.L2], y_train[L.C1][Label.L2], C=1000)
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
    model = svm.SVC(kernel="linear")
    model.fit(X_train[L.C2][Label.L3], y_train[L.C2][Label.L3])
    save_model(model, "c2_label_3_before")
else:
    model = load_model("c2_label_3_before")
predict(model, X_valid[L.C2][Label.L3], y_valid[L.C2][Label.L3])
y_pred_before[L.C2][Label.L3] = model.predict(X_test[L.C2][Label.L3].drop(ID, axis=1))


# In[ ]:


# C2, L4

if RETRAIN:
    model = svm.SVC(kernel="linear", class_weight="balanced")
    model.fit(X_train[L.C2][Label.L4], y_train[L.C2][Label.L4])
    save_model(model, "c2_label_4_before")
else:
    model = load_model("c2_label_4_before")
predict(model, X_valid[L.C2][Label.L4], y_valid[L.C2][Label.L4])
y_pred_before[L.C2][Label.L4] = model.predict(X_test[L.C2][Label.L4].drop(ID, axis=1))


# ## With feature engineering

# ### Feature engineering functions

# In[15]:


from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif


def fit_and_transform_pca(*X: pd.DataFrame):
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RNG)
    pca.fit(X[0])
    X_trf = list(map(lambda x: pd.DataFrame(pca.transform(x)), X))
    print("Shape after PCA:", X_trf[0].shape)
    return pca, *X_trf


def univariate_feature_selection(X: pd.DataFrame, y: pd.Series, feature_count=30):
    FROM_MODEL = False
    if FROM_MODEL:
        clf = ExtraTreesClassifier(n_estimators=50, random_state=RNG)
        clf = clf.fit(X, y)
        selector = SelectFromModel(clf, prefit=True)
    else:
        score_func = f_classif
        selector = SelectKBest(score_func, k=feature_count)
        selector = selector.fit(X, y)
    X_new = selector.transform(X)
    print("Shape after univariate:", X_new.shape)
    return selector, X_new


def combine_transformers(*transformers):
    def combined_transform(X):
        for transformer in transformers:
            X = transformer.transform(X)
        return X

    return combined_transform


def transform(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    pca_count=5,
    feature_drop=0,
):
    if pca_count > 0:
        log("Running PCA...")
        transformers = [None for _ in range(pca_count)]
        transformers[0], X_train_trf, X_valid_trf, X_test_trf = fit_and_transform_pca(
            X_train, X_valid, X_test
        )
    else:
        log("Skipping PCA...")
        transformers = []
        X_train_trf, X_valid_trf, X_test_trf = X_train, X_valid, X_test

    # Skip univariate feature selection if `feature_drop` is specified as 0
    if feature_drop != 0:
        log("Running univariate feature selection...")
        current_feature_count = X_train_trf.shape[1]
        selector, X_train_trf = univariate_feature_selection(
            X_train_trf,
            y_train,
            feature_count=current_feature_count - feature_drop,
        )
        X_valid_trf = pd.DataFrame(selector.transform(X_valid_trf))
        X_test_trf = pd.DataFrame(selector.transform(X_test_trf))
        transformers.append(selector)

    if pca_count > 1:
        log(f"Running PCA {pca_count - 1} times...")
        for i in range(pca_count - 1):
            (
                transformers[i + 1],
                X_train_trf,
                X_valid_trf,
                X_test_trf,
            ) = fit_and_transform_pca(X_train_trf, X_valid_trf, X_test_trf)

    return X_train_trf, X_valid_trf, X_test_trf, combine_transformers(*transformers)


# ### Competition 1 (layer 7)

# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C1][Label.L1],
    y_train[L.C1][Label.L1],
    X_valid[L.C1][Label.L1],
    X_test[L.C1][Label.L1].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    param_grid = {
        "iterations": [100, 200, 300],
        "loss_function": ["MultiClass"],
        "max_depth": range(4, 10, 2),
    }
    # model = svm.SVC(kernel="rbf", C=100, gamma=0.0001, random_state=RNG)
    model = CatBoostClassifier(iterations=100, depth=6, random_state=RNG_SEED)
    # tune(model, X_train_trf, y_train[L.C1][Label.L1], param_grid=param_grid)
    log("Training...")
    model.fit(X_train_trf, y_train[L.C1][Label.L1])
    save_model(model, "c1_label_1_after")
else:
    model = load_model("c1_label_1_after")
predict(model, X_valid_trf, y_valid[L.C1][Label.L1])
y_pred_after[L.C1][Label.L1] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C1][Label.L2],
    y_train[L.C1][Label.L2],
    X_valid[L.C1][Label.L2],
    X_test[L.C1][Label.L2].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    model = svm.SVC(kernel="rbf", C=1000, gamma=0.0001, random_state=RNG, verbose=True)
    # tune(model, X_train_trf, y_train[L.C1][Label.L3])
    log("Training...")
    model.fit(X_train_trf, y_train[L.C1][Label.L2])
    # save_model(model, "c1_label_2_after")
else:
    model = load_model("c1_label_2_after")
predict(model, X_valid_trf, y_valid[L.C1][Label.L2])
y_pred_after[L.C1][Label.L2] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C1][Label.L3],
    y_train[L.C1][Label.L3],
    X_valid[L.C1][Label.L3],
    X_test[L.C1][Label.L3].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    # model = svm.SVC(kernel="rbf", C=100, gamma='scale', random_state=RNG, verbose=True)
    model = svm.SVC(kernel="rbf", random_state=RNG, verbose=True)
    # tune(model, X_train_trf, y_train[L.C1][Label.L3])
    log("Training...")
    model.fit(X_train_trf, y_train[L.C1][Label.L3])
    save_model(model, "c1_label_3_after")
else:
    model = load_model("c1_label_3_after")
predict(model, X_valid_trf, y_valid[L.C1][Label.L3])
y_pred_after[L.C1][Label.L3] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C1][Label.L4],
    y_train[L.C1][Label.L4],
    X_valid[L.C1][Label.L4],
    X_test[L.C1][Label.L4].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    model = svm.SVC(kernel="rbf", class_weight="balanced", random_state=RNG, verbose=True)
    log("Training...")
    model.fit(X_train_trf, y_train[L.C1][Label.L4])
    save_model(model, "c1_label_4_after")
else:
    model = load_model("c1_label_4_after")
predict(model, X_valid_trf, y_valid[L.C1][Label.L4])
y_pred_after[L.C1][Label.L4] = model.predict(X_test_trf)


# ### Competition 2 (layer 12)

# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C2][Label.L1],
    y_train[L.C2][Label.L1],
    X_valid[L.C2][Label.L1],
    X_test[L.C2][Label.L1].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    # model = svm.SVC(kernel="rbf", C=1000, gamma="scale", random_state=RNG)
    model = CatBoostClassifier(random_state=RNG_SEED)
    # param_grid = {
    #     "iterations": [100, 200, 300],
    #     "depth": [4, 6, 8],
    # }
    # tune(model, X_train_trf, y_train[L.C2][Label.L1], param_grid=param_grid)
    log("Training...")
    model.fit(X_train_trf, y_train[L.C2][Label.L1])
    save_model(model, "c2_label_1_after")
else:
    model = load_model("c2_label_1_after")
predict(model, X_valid_trf, y_valid[L.C2][Label.L1])
y_pred_after[L.C2][Label.L1] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C2][Label.L2],
    y_train[L.C2][Label.L2],
    X_valid[L.C2][Label.L2],
    X_test[L.C2][Label.L2].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)

if RETRAIN:
    model = CatBoostClassifier(random_state=RNG_SEED)
    # model = svm.SVC(kernel="rbf", C=100, gamma='scale', random_state=RNG)
    # tune(model, X_train_trf, y_train[L.C2][Label.L2])
    log("Training...")
    model.fit(X_train_trf, y_train[L.C2][Label.L2])
    save_model(model, "c2_label_2_after")
else:
    model = load_model("c2_label_2_after")
predict(model, X_valid_trf, y_valid[L.C2][Label.L2])
y_pred_after[L.C2][Label.L2] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C2][Label.L3],
    y_train[L.C2][Label.L3],
    X_valid[L.C2][Label.L3],
    X_test[L.C2][Label.L3].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    param_grid = {
        "iterations": [100, 200, 300],
        "loss_function": ["MultiClass"],
        "max_depth": range(4, 10, 2),
    }
    model = CatBoostClassifier(iterations=300, loss_function="MultiClass", max_depth=6, random_state=RNG_SEED)
    # model = svm.SVC(kernel="rbf", C=10, random_state=RNG)
    # tune(model, X_train_trf, y_train[L.C2][Label.L3], param_grid=param_grid)
    log("Training...")
    model.fit(X_train_trf, y_train[L.C2][Label.L3])
    save_model(model, "c2_label_3_after")
else:
    model = load_model("c2_label_3_after")
predict(model, X_valid_trf, y_valid[L.C2][Label.L3])
y_pred_after[L.C2][Label.L3] = model.predict(X_test_trf)


# In[ ]:


X_train_trf, X_valid_trf, X_test_trf, selector = transform(
    X_train[L.C2][Label.L4],
    y_train[L.C2][Label.L4],
    X_valid[L.C2][Label.L4],
    X_test[L.C2][Label.L4].drop(ID, axis=1),
    pca_count=1,
    feature_drop=0,
)
if RETRAIN:
    # model = svm.SVC(kernel="rbf", C=100, class_weight="balanced", verbose=True, random_state=RNG)
    model = CatBoostClassifier(random_state=RNG_SEED)
    # tune(model, X_train_trf, y_train[L.C2][Label.L4])
    log("Training...")
    model.fit(X_train_trf, y_train[L.C2][Label.L4])
    save_model(model, "c2_label_4_after")
else:
    model = load_model("c2_label_4_after")
predict(model, X_valid_trf, y_valid[L.C2][Label.L4])
y_pred_after[L.C2][Label.L4] = model.predict(X_test_trf)


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


# In[ ]:




