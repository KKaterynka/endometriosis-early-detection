"""
This module supports multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, AdaBoost, MLP),
model evaluation metrics, cross-validation and SHAP-based model explainability.
"""

import pandas as pd
import numpy as np

import sys
import contextlib

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging

logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
)
import shap


def evaluate_model(y_true, y_pred):
    """
    Evaluate classification model predictions using multiple metrics.

    Parameters:
        y_true: True class labels.
        y_pred: Predicted class labels.

    Returns:
        confusion_matrix, accuracy, recall, specificity, f1_score
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    return cm, acc, recall, specificity, f1


def print_confusion_metrics(cm, accuracy, recall, specificity, f1):
    """
    Print confusion matrix and evaluation metrics.

    Parameters:
        cm: Confusion matrix.
        accuracy: Accuracy score.
        recall: Recall score.
        specificity: Specificity score.
        f1: F1 score.
    """
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")


def run_baseline_logistic_reg(X_train, y_train, cv_folds=5):
    """
    Fit a baseline logistic regression using statsmodels with cross-validation.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of cross-validation folds.

    Returns:
        Fitted model on full training data.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = sm.Logit(y_train_fold, X_train_fold)
        result = model.fit(disp=False)

        y_val_pred = (result.predict(X_val_fold) >= 0.5).astype(int)
        _, acc, recall, specificity, f1 = evaluate_model(y_val_fold, y_val_pred)
        scores.append([acc, recall, specificity, f1])

    scores = np.array(scores)
    print(f"Average F1-score across folds: {scores[:, 3].mean()}")

    final_model = sm.Logit(y_train, X_train)
    final_result = final_model.fit(disp=False)

    return final_result


def run_logistic_regression(X_train, y_train, suppress_logs=True, disp=False):
    """
    Run logistic regression with hyperparameter tuning via GridSearchCV.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        suppress_logs: Suppress verbose outputs.
        disp: Whether to display best parameters and F1 score.

    Returns:
        Best fitted model and best F1 score.
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_iter = 10000

    @contextlib.contextmanager
    def suppress_output():
        if suppress_logs:
            with open("/dev/null", "w") as fnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = fnull
                try:
                    yield
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
        else:
            yield

    param_grids = [
        {
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "C": [0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9],
        },
    ]

    best_model, best_score, best_params = None, 0, None
    with suppress_output():
        for grid in param_grids:
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=max_iter),
                param_grid=grid,
                scoring="f1",
                cv=folds,
                verbose=1,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            if grid_search.best_score_ > best_score:
                best_model, best_score, best_params = (
                    grid_search.best_estimator_,
                    grid_search.best_score_,
                    grid_search.best_params_,
                )

    if disp:
        print(f"Best Hyperparameters: {best_params}\nAvg F1 Score: {best_score}")

    best_model.fit(X_train, y_train)

    return best_model, best_score


def run_model(X_train, y_train, model_cls, param_grid, suppress_logs=True, disp=False):
    """
    General model training function with hyperparameter tuning.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        model_cls: Classifier class (e.g., RandomForestClassifier).
        param_grid: Parameters to tune.
        suppress_logs: Suppress verbose output.
        disp: Display best parameters and F1 score.

    Returns:
        Best trained model and best F1 score.
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    @contextlib.contextmanager
    def suppress_output():
        if suppress_logs:
            with open("/dev/null", "w") as fnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = fnull
                try:
                    yield
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
        else:
            yield

    model = model_cls(random_state=42)

    with suppress_output():
        grid_search = GridSearchCV(
            model, param_grid, scoring="f1", cv=folds, verbose=0, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

    if disp:
        print(
            f"Best Hyperparameters: {grid_search.best_params_}\nAvg F1 Score: {grid_search.best_score_}"
        )

    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train)

    return final_model, grid_search.best_score_


# hyperparameter grids for Random Forest, XGBoost and AdaBoost
param_grids = {
    "rf": {
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [4, 6, 8],
        "min_samples_split": [2, 4, 6],
        "n_estimators": [100, 200, 300],
    },
    "xgb": {
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.5, 0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
        "gamma": [0.05, 0.1],
    },
    "ada": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "algorithm": ["SAMME"],
    },
}


def run_rf(X_train, y_train, disp=False):
    """
    Trains a Random Forest classifier with GridSearchCV on the training data.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        disp: Whether to display the best parameters and F1 score.

    Returns:
        Trained model, best F1 score from cross-validation.
    """
    return run_model(
        X_train, y_train, RandomForestClassifier, param_grids["rf"], disp=disp
    )


def run_xgb(X_train, y_train, disp=False):
    """
    Trains an XGBoost classifier with GridSearchCV on the training data.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        disp: Whether to display the best parameters and F1 score.

    Returns:
        Trained model, best F1 score from cross-validation.
    """
    return run_model(X_train, y_train, XGBClassifier, param_grids["xgb"], disp=disp)


def run_ada(X_train, y_train, disp=False):
    """
    Trains an AdaBoost classifier with GridSearchCV on the training data.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        disp: Whether to display the best parameters and F1 score.

    Returns:
        Trained model, best F1 score from cross-validation.
    """
    return run_model(
        X_train, y_train, AdaBoostClassifier, param_grids["ada"], disp=disp
    )


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluates the given model on test data and prints performance metrics.

    Parameters:
        model: Trained classifier.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Predictions on test data.
    """
    y_pred = model.predict(X_test)
    cm, acc, recall, specificity, f1 = evaluate_model(y_test, y_pred)
    print_confusion_metrics(cm, acc, recall, specificity, f1)
    return y_pred


def run_mlp(X, y, cv=5, param_grid=None, disp=False, pu_learning=False):
    """
    Trains an MLPClassifier using a pipeline and performs hyperparameter tuning with GridSearchCV.

    Parameters:
        X: Input features.
        y: Labels.
        cv: Number of cross-validation folds.
        param_grid: Custom hyperparameter grid.
        disp: Whether to print the best parameters and F1 score.
        pu_learning: Whether to use smaller batch sizes for PU-learning.

    Returns:
        Best trained model and corresponding F1 score from cross-validation.
    """
    if param_grid is None:
        param_grid = {
            "mlp__hidden_layer_sizes": [
                (50,),
                (100,),
                (50, 50),
                (100, 50),
                (100, 100, 50),
            ],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [0.0001, 0.001, 0.01, 0.1],
            "mlp__solver": ["adam"],
            "mlp__learning_rate_init": [0.001, 0.01, 0.1],
            "mlp__batch_size": [32, 64, 128],
            "mlp__beta_1": [0.9, 0.95, 0.99],
            "mlp__beta_2": [0.999, 0.9999],
            "mlp__early_stopping": [True],
            "mlp__max_iter": [500, 1000],
        }
        if pu_learning:
            param_grid["mlp__batch_size"] = [1, 8, 32]

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("mlp", MLPClassifier(random_state=42))]
    )

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring="f1")
    grid_search.fit(X, y)

    if disp:
        print(
            f"Best Hyperparameters: {grid_search.best_params_}\nAvg F1 Score: {grid_search.best_score_}"
        )

    final_model = grid_search.best_estimator_
    final_model.fit(X, y)

    return final_model, grid_search.best_score_


def run_models_on_top_features(
    X_train,
    y_train,
    X_test,
    top_features,
    min_features=None,
    max_features=None,
):
    """
    Trains multiple classifiers on subsets of top features and selects the best model.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        top_features: Ranked list of feature names.
        min_features (optional, defaults to 5): Minimum number of features to start with.
        max_features (optional, defaults to maximum number of features): Maximum number of top features to try.

    Returns:
        best_model: The best performing model.
        best_X_test: Test set with selected features.
        best_f1: Best F1 score achieved.
        Lists of F1 scores for each model type.
    """
    models = {
        "Logistic Regression": (run_logistic_regression, []),
        "Random Forest": (run_rf, []),
        "XGBoost": (run_xgb, []),
        "ADABoost": (run_ada, []),
        "MLP": (run_mlp, []),
    }

    best_model, best_X_test, best_f1 = None, None, None

    min_features = min_features if min_features else 5
    max_features = max_features if max_features else len(top_features)

    for n_top_features in range(5, max_features, 1):
        if isinstance(top_features, pd.DataFrame) and "feature" in top_features.columns:
            curr_features = top_features["feature"][:n_top_features].values
        else:
            curr_features = top_features[:n_top_features]
        curr_features = list(curr_features)
        X_train_selected = X_train[list(curr_features)]
        X_test_selected = X_test[list(curr_features)]

        for model_name, (model_func, f1_scores) in models.items():
            model_args = (X_train_selected, y_train)
            curr_model, avg_f1 = model_func(*model_args)

            f1_scores.append(avg_f1)
            if best_f1 is None or avg_f1 > best_f1:
                best_model, best_X_test, best_f1 = curr_model, X_test_selected, avg_f1

    return (
        best_model,
        best_X_test,
        best_f1,
        *(scores for _, scores, *_ in models.values()),
    )


def get_shap_importance(X_train, model):
    """
    Computes SHAP feature importance for a trained model.

    Parameters:
        X_train: Training features.
        model: Trained classifier.

    Returns:
        SHAP values for each feature.
        Mean absolute SHAP importance per feature.
    """
    if "Ada" in str(type(model)) or "MLP" in str(type(model)):
        explainer = shap.Explainer(model.predict_proba, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    if len(mean_shap_values.shape) > 1:
        mean_shap_values = mean_shap_values[:, 1]

    shap_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": mean_shap_values}
    ).sort_values(by="importance", ascending=False)

    return shap_values, shap_importance
