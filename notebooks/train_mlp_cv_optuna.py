# """
# Training module: MLP + Optuna with CV on training, validation-driven optimization,
# and comprehensive evaluation and plots for train/val/test.

# Usage (inside a notebook):

# from src.train_mlp_cv_optuna import run_training
# results = run_training(X, y, n_trials=20, random_state=42, cv_folds=4, verbose=True)

# This will:
# - Split data like your notebook (80% per class to train; remaining split equally into val/test)
# - Run Optuna minimizing validation log loss while performing CV on the training set
# - Train best model and report metrics + plots on train, val, and test
# - Refit on train+val and report final test metrics + plots

# Returned `results` contains:
# {
#   'study': optuna.study.Study,
#   'best_params': dict,
#   'splits': { 'train_idx', 'val_idx', 'test_idx' },
#   'metrics': {
#       'train': {...}, 'val': {...}, 'test': {...},
#       'final_test_refit_trainval': {...}
#   }
# }

# Note: Copy this file or its functions into your notebook as needed.
# """

# from __future__ import annotations

# import warnings
# from dataclasses import dataclass
# from typing import Dict, Tuple, Any, Optional

# import numpy as np
# import optuna
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     roc_curve,
#     auc,
#     precision_recall_curve,
#     average_precision_score,
#     log_loss,
# )

# warnings.filterwarnings("ignore")


# # -----------------------------
# # Data splitting utility
# # -----------------------------

# def stratified_custom_split_indices(y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Create balanced splits with class parity using a 70/15/15 rule.
#     Each split (train/val/test) will have the SAME number of positives and negatives.
#     We base counts on the minority class to ensure feasibility.

#     Returns (train_idx, val_idx, test_idx)
#     """
#     y_flat = y.ravel()

#     pos_idx = np.where(y_flat == 1)[0]
#     neg_idx = np.where(y_flat == 0)[0]

#     n_pos = len(pos_idx)
#     n_neg = len(neg_idx)
#     n_min = min(n_pos, n_neg)

#     # Target per-class counts using 70/15/15 split. Work with integers.
#     train_per_class = int(np.floor(0.70 * n_min))
#     val_per_class = int(np.floor(0.15 * n_min))
#     test_per_class = int(np.floor(0.15 * n_min))

#     # Adjust for any rounding remainder by assigning to train (keeps 70% dominant)
#     remainder = n_min - (train_per_class + val_per_class + test_per_class)
#     train_per_class += remainder

#     rng = np.random.RandomState(random_state)

#     # Sample per class without replacement
#     pos_perm = rng.permutation(pos_idx)
#     neg_perm = rng.permutation(neg_idx)

#     pos_train = pos_perm[:train_per_class]
#     pos_val = pos_perm[train_per_class:train_per_class + val_per_class]
#     pos_test = pos_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

#     neg_train = neg_perm[:train_per_class]
#     neg_val = neg_perm[train_per_class:train_per_class + val_per_class]
#     neg_test = neg_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

#     # Combine splits and shuffle indices within each split for randomness
#     train_idx = np.concatenate([pos_train, neg_train])
#     val_idx = np.concatenate([pos_val, neg_val])
#     test_idx = np.concatenate([pos_test, neg_test])

#     rng.shuffle(train_idx)
#     rng.shuffle(val_idx)
#     rng.shuffle(test_idx)

#     return train_idx, val_idx, test_idx


# # -----------------------------
# # Model/pipeline and objective
# # -----------------------------

# def build_pipeline(trial: optuna.Trial, random_state: int) -> Pipeline:
#     """Create a StandardScaler + MLPClassifier pipeline with hyperparameters from Optuna."""
#     hidden_layer_sizes = []
#     n_layers = trial.suggest_int("n_layers", 1, 5)
#     for i in range(n_layers):
#         layer_size = trial.suggest_int(f"layer_{i}_size", 50, 500)
#         hidden_layer_sizes.append(layer_size)

#     activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])  # logistic = sigmoid
#     solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])  # lbfgs often strong on smaller datasets
#     alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
#     learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True)

#     mlp = MLPClassifier(
#         hidden_layer_sizes=tuple(hidden_layer_sizes),
#         activation=activation,
#         solver=solver,
#         alpha=alpha,
#         learning_rate_init=learning_rate_init,
#         max_iter=1000,
#         early_stopping=True,
#         validation_fraction=0.1,
#         n_iter_no_change=20,
#         random_state=random_state,
#     )

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("mlp", mlp),
#     ])
#     return pipe


# def objective(
#     trial: optuna.Trial,
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     random_state: int,
#     cv_folds: int,
# ) -> float:
#     """Optuna objective that:
#     - Builds a pipeline
#     - Runs CV (accuracy) on training data for robustness signal
#     - Fits on full training data
#     - Computes validation log loss and returns it to be MINIMIZED
#     """
#     pipe = build_pipeline(trial, random_state)

#     # CV on training set (does not see validation)
#     cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
#     cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
#     mean_cv_acc = float(np.mean(cv_scores))

#     # Fit on full training, evaluate on validation
#     pipe.fit(X_train, y_train)

#     # Use probabilities for log loss and ROC/PR later
#     y_val_proba = pipe.predict_proba(X_val)[:, 1]
#     val_logloss = float(log_loss(y_val, y_val_proba))
#     val_acc = float(accuracy_score(y_val, (y_val_proba >= 0.5).astype(int)))

#     # Store extra info for analysis
#     trial.set_user_attr("mean_cv_acc", mean_cv_acc)
#     trial.set_user_attr("val_acc_from_best_fit", val_acc)

#     return val_logloss  # MINIMIZE validation error (log loss)


# # -----------------------------
# # Evaluation helpers
# # -----------------------------

# @dataclass
# class EvalResult:
#     accuracy: float
#     logloss: float
#     report: str


# def evaluate_split(pipe: Pipeline, X: np.ndarray, y: np.ndarray, title_prefix: str, plot: bool = True) -> EvalResult:
#     y_proba = pipe.predict_proba(X)[:, 1]
#     y_pred = (y_proba >= 0.5).astype(int)

#     acc = accuracy_score(y, y_pred)
#     ll = log_loss(y, y_proba)
#     rep = classification_report(y, y_pred, digits=4)

#     if plot:
#         plot_all(y_true=y, y_pred=y_pred, y_proba=y_proba, title_prefix=title_prefix)

#     return EvalResult(accuracy=acc, logloss=ll, report=rep)


# def plot_all(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, title_prefix: str) -> None:
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
#     axes[0].set_title(f"{title_prefix} - Confusion Matrix")
#     axes[0].set_xlabel("Predicted")
#     axes[0].set_ylabel("True")

#     # ROC Curve
#     fpr, tpr, _ = roc_curve(y_true, y_proba)
#     roc_auc = auc(fpr, tpr)
#     axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
#     axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
#     axes[1].set_title(f"{title_prefix} - ROC Curve")
#     axes[1].set_xlabel("False Positive Rate")
#     axes[1].set_ylabel("True Positive Rate")
#     axes[1].legend(loc="lower right")

#     # Precision-Recall Curve
#     precision, recall, _ = precision_recall_curve(y_true, y_proba)
#     ap = average_precision_score(y_true, y_proba)
#     axes[2].plot(recall, precision, label=f"AP = {ap:.4f}")
#     axes[2].set_title(f"{title_prefix} - Precision-Recall")
#     axes[2].set_xlabel("Recall")
#     axes[2].set_ylabel("Precision")
#     axes[2].legend(loc="lower left")

#     plt.tight_layout()
#     plt.show()


# # -----------------------------
# # Orchestrator
# # -----------------------------

# def run_training(
#     X: np.ndarray,
#     y: np.ndarray,
#     n_trials: int = 2,
#     random_state: int = 42,
#     cv_folds: int = 4,
#     verbose: bool = True,
#     save_best_model_path: Optional[str] = None,
#     save_study_path: Optional[str] = None,
#     save_final_model_path: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """Run the full training procedure described.

#     - Custom stratified split (80% per class -> train; rest -> val/test)
#     - Optuna minimizing validation log loss; CV on training set
#     - Evaluate best model on train, val, test
#     - Refit on train+val and evaluate on test again (final)
#     """
#     # 1) Create splits
#     train_idx, val_idx, test_idx = stratified_custom_split_indices(y, random_state)
#     X_train, y_train = X[train_idx], y[train_idx].ravel()
#     X_val, y_val = X[val_idx], y[val_idx].ravel()
#     X_test, y_test = X[test_idx], y[test_idx].ravel()

#     if verbose:
#         from collections import Counter
#         print("train shape:", X_train.shape, "class counts:", Counter(y_train))
#         print("val   shape:", X_val.shape,   "class counts:", Counter(y_val))
#         print("test  shape:", X_test.shape,  "class counts:", Counter(y_test))

#     # 2) Optuna study: minimize validation log loss
#     study = optuna.create_study(direction="minimize")

#     if verbose:
#         print(f"Iniciando otimização com {n_trials} trials... (direção: minimizar log loss de validação)")

#     study.optimize(
#         lambda trial: objective(trial, X_train, y_train, X_val, y_val, random_state, cv_folds),
#         n_trials=n_trials,
#         show_progress_bar=verbose,
#     )

#     if verbose:
#         print("Otimização concluída!")
#         print(f"Melhor valor (val log loss): {study.best_value:.6f}")
#         print(f"Melhores parâmetros: {study.best_params}")
#     # Optional: persist the study
#     if save_study_path is not None:
#         with open(save_study_path, "wb") as f:
#             pickle.dump(study, f)

#     # 3) Train best model on TRAIN only, evaluate on train/val/test
#     best_pipe = build_pipeline(study.best_trial, random_state)
#     best_pipe.fit(X_train, y_train)
#     # Optional: persist best model (trained on TRAIN only)
#     if save_best_model_path is not None:
#         with open(save_best_model_path, "wb") as f:
#             pickle.dump(best_pipe, f)

#     if verbose:
#         print("\nAvaliação com melhor modelo (treinado apenas em TRAIN):")

#     train_eval = evaluate_split(best_pipe, X_train, y_train, title_prefix="Train")
#     if verbose:
#         print("[TRAIN] accuracy=%.4f logloss=%.6f" % (train_eval.accuracy, train_eval.logloss))
#         print(train_eval.report)

#     val_eval = evaluate_split(best_pipe, X_val, y_val, title_prefix="Validação")
#     if verbose:
#         print("[VAL]   accuracy=%.4f logloss=%.6f" % (val_eval.accuracy, val_eval.logloss))
#         print(val_eval.report)

#     test_eval = evaluate_split(best_pipe, X_test, y_test, title_prefix="Teste (modelo treinado em TRAIN)")
#     if verbose:
#         print("[TEST]  accuracy=%.4f logloss=%.6f" % (test_eval.accuracy, test_eval.logloss))
#         print(test_eval.report)

#     # 4) Refit on TRAIN+VAL and evaluate on TEST (final report)
#     X_trval = np.vstack([X_train, X_val])
#     y_trval = np.concatenate([y_train, y_val])

#     final_pipe = build_pipeline(study.best_trial, random_state)
#     final_pipe.fit(X_trval, y_trval)
#     # Optional: persist final refit model (TRAIN+VAL)
#     if save_final_model_path is not None:
#         with open(save_final_model_path, "wb") as f:
#             pickle.dump(final_pipe, f)

#     if verbose:
#         print("\nAvaliação FINAL no TESTE (modelo treinado em TRAIN+VAL):")

#     final_test_eval = evaluate_split(final_pipe, X_test, y_test, title_prefix="Teste FINAL (modelo treinado em TRAIN+VAL)")
#     if verbose:
#         print("[FINAL TEST] accuracy=%.4f logloss=%.6f" % (final_test_eval.accuracy, final_test_eval.logloss))
#         print(final_test_eval.report)

#     return {
#         "study": study,
#         "best_params": study.best_params,
#         "splits": {
#             "train_idx": train_idx,
#             "val_idx": val_idx,
#             "test_idx": test_idx,
#         },
#         "metrics": {
#             "train": {
#                 "accuracy": train_eval.accuracy,
#                 "logloss": train_eval.logloss,
#                 "report": train_eval.report,
#             },
#             "val": {
#                 "accuracy": val_eval.accuracy,
#                 "logloss": val_eval.logloss,
#                 "report": val_eval.report,
#             },
#             "test": {
#                 "accuracy": test_eval.accuracy,
#                 "logloss": test_eval.logloss,
#                 "report": test_eval.report,
#             },
#             "final_test_refit_trainval": {
#                 "accuracy": final_test_eval.accuracy,
#                 "logloss": final_test_eval.logloss,
#                 "report": final_test_eval.report,
#             },
#         },
#     }


# # if __name__ == "__main__":
# #     results = run_training(
# #         X, y,
# #         n_trials=20,
# #         random_state=42,
# #         cv_folds=4,
# #         verbose=True,
# #         # Optional persistence (uncomment to save):
# #         save_best_model_path="artifacts/best_model_train.pkl",
# #         save_final_model_path="artifacts/final_model_trainval.pkl",
# #         save_study_path="artifacts/optuna_study.pkl",
# #     )

# %% [markdown]
# ==============================
#  Utilitários de Treino por Época (MLPClassifier)
#  - Split 70/15/15 estratificado
#  - Treino com partial_fit em épocas e mini-batches
#  - Métricas por época (loss/accuracy de treino e validação)
#  - Early stopping com "patience" + restauração de melhores pesos
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

from sklearn.metrics import accuracy_score, log_loss, f1_score
import itertools

import numpy as np
import pandas as pd
from typing import Tuple

# -----------------------------
# Split balanceado 70/15/15 (por índices)
# -----------------------------
def stratified_custom_split_indices(
    y: np.ndarray,
    random_state: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria splits balanceados (binário 0/1) com 70/15/15.
    Cada split terá o MESMO número de positivos e negativos.
    Baseia-se na classe minoritária; amostras excedentes da majoritária ficam de fora.

    Retorna (train_idx, val_idx, test_idx) como índices POSICIONAIS (0..n-1).
    """
    y_flat = np.asarray(y).ravel()

    # checagens básicas
    uniq = np.unique(y_flat)
    if len(uniq) != 2:
        raise ValueError(f"Esta função pressupõe problema binário (2 classes). Encontradas: {uniq}.")

    # assume 1 como positivo e 0 como negativo (compatível com seu código)
    pos_idx = np.where(y_flat == 1)[0]
    neg_idx = np.where(y_flat == 0)[0]

    n_min = min(len(pos_idx), len(neg_idx))
    if n_min < 2:
        raise ValueError("Poucas amostras na classe minoritária para formar splits úteis.")

    # contas por classe (inteiros)
    train_per_class = int(np.floor(train_frac * n_min))
    val_per_class   = int(np.floor(val_frac   * n_min))
    test_per_class  = int(np.floor(test_frac  * n_min))

    # garante soma igual à minoria (joga resto para treino)
    remainder = n_min - (train_per_class + val_per_class + test_per_class)
    train_per_class += remainder

    rng = np.random.RandomState(random_state)
    pos_perm = rng.permutation(pos_idx)
    neg_perm = rng.permutation(neg_idx)

    pos_train = pos_perm[:train_per_class]
    pos_val   = pos_perm[train_per_class:train_per_class + val_per_class]
    pos_test  = pos_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

    neg_train = neg_perm[:train_per_class]
    neg_val   = neg_perm[train_per_class:train_per_class + val_per_class]
    neg_test  = neg_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

    train_idx = np.concatenate([pos_train, pos_train*0 + neg_train])  # concat pos/neg
    val_idx   = np.concatenate([pos_val,   pos_val*0   + neg_val])
    test_idx  = np.concatenate([pos_test,  pos_test*0  + neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # sanidade: não sobrepor
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    return train_idx, val_idx, test_idx


# -----------------------------
# Conveniência: devolve X/y já fatiados (70/15/15 balanceado)
# -----------------------------
def split_70_15_15_balanced(X, y, random_state: int = 42):
    """
    Aplica 'stratified_custom_split_indices' e retorna:
    X_train, X_val, X_test, y_train, y_val, y_test
    Funciona com DataFrame/Series e também com arrays NumPy.
    """
    y_arr = np.asarray(y).ravel()
    tr_idx, va_idx, te_idx = stratified_custom_split_indices(y_arr, random_state=random_state)

    def _take(a, idx):
        if isinstance(a, (pd.DataFrame, pd.Series)):
            return a.iloc[idx]
        return a[idx]

    X_train, X_val, X_test = _take(X, tr_idx), _take(X, va_idx), _take(X, te_idx)
    y_train, y_val, y_test = _take(y, tr_idx), _take(y, va_idx), _take(y, te_idx)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------- Mini-batches ----------
from dataclasses import dataclass

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    train_f1: float
    val_f1: float


def _iter_minibatches(X, y, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size]
        yield X[sl], np.asarray(y)[sl]

def _build_preprocessor(
    X_train,
    *,
    categorical_vars: Optional[List[str]] = None,
    continuous_vars: Optional[List[str]] = None,
    discrete_vars: Optional[List[str]] = None,
):
    """
    Cria e 'fit' um ColumnTransformer para X_train usando as listas explícitas de colunas:
      - Numéricos (contínuos + discretos): Imputer(median) + StandardScaler
      - Categóricos: Imputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    """
    if not isinstance(X_train, pd.DataFrame):
        # Se vier NumPy, aplica tudo como numérico
        num_features = list(range(X_train.shape[1]))
        pre = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), num_features)
            ],
            remainder='drop'
        )
        pre.fit(X_train)
        return pre

    # Garante que só use colunas existentes (robustez)
    cat = [c for c in (categorical_vars or []) if c in X_train.columns]
    cont = [c for c in (continuous_vars or []) if c in X_train.columns]
    disc = [c for c in (discrete_vars or []) if c in X_train.columns]
    num = cont + disc

    transformers = []
    if num:
        transformers.append(
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), num)
        )
    if cat:
        # Compat com sklearn 1.0–1.5 (sparse) e 1.2+ (sparse_output)
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            # fallback para versões antigas (onde 'sparse_output' não existe)
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        transformers.append(
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', ohe),
            ]), cat)
        )


    if not transformers:
        raise ValueError("Nenhuma coluna válida encontrada para pré-processar.")

    pre = ColumnTransformer(transformers=transformers, remainder='drop')
    pre.fit(X_train)
    return pre



# ---------- Treino com métricas por época ----------
def train_mlp_with_metrics(
    X_train, y_train, X_val, y_val,
    *,
    categorical_vars=None,
    continuous_vars=None,
    discrete_vars=None,
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=64,
    max_epochs=200,
    patience=20,
    tol=1e-4,
    random_state=42,
    shuffle=True,
    monitor: str = "val_f1",
    monitor_mode: str = "max",
    # >>> novos (opcionais, usados p/ solver='sgd'):
    momentum: float | None = None,
    nesterovs_momentum: bool | None = None,
    learning_rate: str | None = None,   # 'constant' | 'invscaling' | 'adaptive'
):
    """
    Treina MLP por épocas (partial_fit) com métricas por época e early stopping
    baseado em `monitor` (ex.: 'val_f1' para maximizar F1 de validação).
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)

    # --- Pré-processamento ---
    pre = _build_preprocessor(
        X_train,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        discrete_vars=discrete_vars,
    )
    X_train_t = pre.transform(X_train)
    X_val_t   = pre.transform(X_val)

    # --- Modelo ---
    clf_params = dict(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
        n_iter_no_change=np.inf,
        tol=0.0,
        early_stopping=False,
    )
    # aplica extras se fornecidos (relevantes p/ SGD)
    if momentum is not None:
        clf_params["momentum"] = momentum
    if nesterovs_momentum is not None:
        clf_params["nesterovs_momentum"] = nesterovs_momentum
    if learning_rate is not None:
        clf_params["learning_rate"] = learning_rate

    clf = MLPClassifier(**clf_params)

    history = []

    # Inicialização do melhor valor conforme monitor
    if monitor_mode == "min":
        best_metric = np.inf
    elif monitor_mode == "max":
        best_metric = -np.inf
    else:
        raise ValueError("monitor_mode deve ser 'min' ou 'max'.")

    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # mini-batches
        n = X_train_t.shape[0]
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            sl = idx[start:start + batch_size]
            Xb, yb = X_train_t[sl], np.asarray(y_train)[sl]
            if epoch == 1 and start == 0:
                clf.partial_fit(Xb, yb, classes=classes)
            else:
                clf.partial_fit(Xb, yb)

        # métricas por época
        p_train = clf.predict_proba(X_train_t)
        p_val   = clf.predict_proba(X_val_t)
        y_pred_train = p_train.argmax(1)
        y_pred_val   = p_val.argmax(1)

        tr_loss = log_loss(y_train, p_train, labels=classes)
        va_loss = log_loss(y_val,   p_val,   labels=classes)
        tr_acc  = accuracy_score(y_train, y_pred_train)
        va_acc  = accuracy_score(y_val,   y_pred_val)

        # F1 binário (pos_label=1). Se der erro (multi-classe), cai p/ macro.
        try:
            tr_f1 = f1_score(y_train, y_pred_train, average="binary", pos_label=1)
            va_f1 = f1_score(y_val,   y_pred_val,   average="binary", pos_label=1)
        except ValueError:
            tr_f1 = f1_score(y_train, y_pred_train, average="macro")
            va_f1 = f1_score(y_val,   y_pred_val,   average="macro")

        history.append(EpochMetrics(epoch, tr_loss, va_loss, tr_acc, va_acc, tr_f1, va_f1))

        # early stopping por monitor
        current = {"val_loss": va_loss, "val_f1": va_f1}.get(monitor)
        if current is None:
            raise ValueError("monitor deve ser 'val_loss' ou 'val_f1'.")

        improved = (
            (monitor_mode == "min" and (best_metric - current) > tol) or
            (monitor_mode == "max" and (current - best_metric) > tol)
        )

        if improved:
            best_metric = current
            best_state = (deepcopy(clf.coefs_), deepcopy(clf.intercepts_))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # restaura melhores pesos
    if best_state is not None:
        coefs, intercepts = best_state
        clf.coefs_ = coefs
        clf.intercepts_ = intercepts

    # --- estatísticas finais ---
    best_epoch_by_val_f1 = max(history, key=(lambda m: m.val_f1)).epoch if history else 0
    best_val_loss = min((m.val_loss for m in history), default=float("nan"))
    final_stats = {
        "epochs_ran": history[-1].epoch if history else 0,
        "best_epoch_by_val_f1": best_epoch_by_val_f1,
        "best_val_f1": max((m.val_f1 for m in history), default=float("nan")),
        "final_val_f1": history[-1].val_f1 if history else float("nan"),
        "final_val_acc": history[-1].val_acc if history else float("nan"),
        "final_val_loss": history[-1].val_loss if history else float("nan"),
        "best_val_loss": float(best_val_loss),
    }
    
    clf.preprocess_ = pre
    return clf, history, final_stats

# ---------- Plot ----------
import matplotlib.pyplot as plt

def plot_history(history: List[EpochMetrics]):
    if not history:
        print("Histórico vazio.")
        return
    epochs = [m.epoch for m in history]
    tr_loss = [m.train_loss for m in history]
    va_loss = [m.val_loss for m in history]
    tr_acc  = [m.train_acc  for m in history]
    va_acc  = [m.val_acc    for m in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, tr_loss, label="Train loss")
    plt.plot(epochs, va_loss, label="Val loss")
    plt.xlabel("Épocas"); plt.ylabel("Log-loss"); plt.title("Curva de perda por época")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, tr_acc, label="Train acc")
    plt.plot(epochs, va_acc, label="Val acc")
    plt.xlabel("Épocas"); plt.ylabel("Acurácia"); plt.title("Acurácia por época")
    plt.legend(); plt.grid(True); plt.show()

import itertools
import pandas as pd

def _as_tuple(h):
    if isinstance(h, int):
        return (h,)
    if isinstance(h, list):
        return tuple(h)
    return tuple(h)

def gridsearch_mlp(
    X_train, y_train, X_val, y_val,
    *,
    categorical_vars=None,
    continuous_vars=None,
    discrete_vars=None,
    param_grid=None,         # dict com listas
    # defaults caso não venham no grid:
    max_epochs=200,
    patience=20,
    tol=1e-4,
    random_state=42,
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=64,
    momentum=None,
    nesterovs_momentum=None,
    learning_rate=None,
    monitor="val_f1",
    monitor_mode="max",
    verbose=1,
):
    """
    Faz grid nas combinações de param_grid. Chaveia por F1 de validação.
    Retorna (best, trials) — compatível com a sua chamada atual.

    Dicas de chaves em param_grid:
    - "hidden_layer_sizes": [ (64,), (64,32), ... ]
    - "activation": ["relu","tanh","logistic"]
    - "solver": ["adam","sgd"]             # 'lbfgs' é ignorado (não suporta partial_fit)
    - "learning_rate_init": [1e-3, 5e-4]   # eta
    - "alpha": [1e-4, 1e-3]
    - "batch_size": [32, 64, 128]
    - "patience": [10, 20, 30]
    - "max_epochs": [200, 300]
    - "tol": [1e-4, 5e-4]
    - "momentum": [0.9]
    - "nesterovs_momentum": [True, False]
    - "learning_rate": ["constant","adaptive"]
    """
    if param_grid is None:
        param_grid = {}

    # cria listas a partir do grid ou dos defaults
    hs_list   = param_grid.get("hidden_layer_sizes", [(64,), (128,), (64,32)])
    act_list  = param_grid.get("activation", ["relu", "tanh"])
    solv_list = param_grid.get("solver", [solver])
    eta_list  = param_grid.get("learning_rate_init", [learning_rate_init])
    alpha_list = param_grid.get("alpha", [alpha])
    bs_list    = param_grid.get("batch_size", [batch_size])
    pat_list   = param_grid.get("patience", [patience])
    mep_list   = param_grid.get("max_epochs", [max_epochs])
    tol_list   = param_grid.get("tol", [tol])
    mom_list   = param_grid.get("momentum", [momentum])
    nes_list   = param_grid.get("nesterovs_momentum", [nesterovs_momentum])
    lrn_list   = param_grid.get("learning_rate", [learning_rate])

    trials = []
    best = None
    trial_id = 0

    # produto cartesiano
    for h, act, solv, eta, a, bs, pat, mep, tl, mom, nes, lrn in itertools.product(
        hs_list, act_list, solv_list, eta_list, alpha_list, bs_list,
        pat_list, mep_list, tol_list, mom_list, nes_list, lrn_list
    ):
        h = _as_tuple(h)

        # 'lbfgs' não suporta partial_fit (sem métricas por época) -> ignoramos
        if str(solv).lower() == "lbfgs":
            if verbose:
                print(f"[GRID][skip] hls={h}, act={act}, solver=lbfgs não suportado para treino por épocas.")
            trials.append({
                "trial_id": trial_id,
                "params": {"hidden_layer_sizes": h, "activation": act, "solver": solv,
                           "learning_rate_init": eta, "alpha": a, "batch_size": bs,
                           "patience": pat, "max_epochs": mep, "tol": tl,
                           "momentum": mom, "nesterovs_momentum": nes, "learning_rate": lrn,
                           "random_state": random_state},
                "stats": {"best_val_f1": float("nan"), "final_val_f1": float("nan"),
                          "final_val_acc": float("nan"), "final_val_loss": float("nan"),
                          "epochs_ran": 0, "best_epoch_by_val_f1": 0, "best_val_loss": float("nan")},
                "history": [],
                "clf": None,
                "status": "skipped_lbfgs",
            })
            trial_id += 1
            continue

        if verbose:
            print(f"[GRID] hls={h}, act={act}, solver={solv}, eta={eta}, alpha={a}, "
                  f"bs={bs}, patience={pat}, max_epochs={mep}, tol={tl}, "
                  f"momentum={mom}, nest={nes}, lr_policy={lrn}")

        clf, hist, stats = train_mlp_with_metrics(
            X_train, y_train, X_val, y_val,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            discrete_vars=discrete_vars,
            hidden_layer_sizes=h,
            activation=act,
            solver=solv,
            alpha=a,
            learning_rate_init=eta,
            batch_size=bs,
            max_epochs=mep,
            patience=pat,
            tol=tl,
            random_state=random_state,
            monitor=monitor,
            monitor_mode=monitor_mode,
            momentum=mom,
            nesterovs_momentum=nes,
            learning_rate=lrn,
        )

        trial = {
            "trial_id": trial_id,
            "params": {"hidden_layer_sizes": h, "activation": act, "solver": solv,
                       "learning_rate_init": eta, "alpha": a, "batch_size": bs,
                       "patience": pat, "max_epochs": mep, "tol": tl,
                       "momentum": mom, "nesterovs_momentum": nes, "learning_rate": lrn,
                       "random_state": random_state},
            "stats": stats,
            "history": hist,
            "clf": clf,
            "status": "ok",
        }
        trials.append(trial)
        trial_id += 1

        score = stats.get("best_val_f1", float("-inf"))
        if (best is None) or (score > best["stats"].get("best_val_f1", float("-inf"))):
            best = trial

    return best, trials

def summarize_trials(trials, save_path: str | None = None) -> pd.DataFrame:
    """Cria um DataFrame resumo de TODOS os ensaios e (opcionalmente) salva em CSV."""
    rows = []
    for t in trials:
        r = {"trial_id": t["trial_id"], "status": t.get("status","ok")}
        # params (como string p/ serializar tuplas)
        for k, v in t["params"].items():
            r[k] = v if not isinstance(v, tuple) else str(v)
        # stats
        for k, v in t["stats"].items():
            r[k] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def export_trials_history(trials, save_path: str | None = None) -> pd.DataFrame:
    """
    Exporta um DataFrame 'long' com as métricas por ÉPOCA para cada trial.
    Útil para análises/plots posteriores.
    """
    rows = []
    for t in trials:
        trial_id = t["trial_id"]
        params = {k: (v if not isinstance(v, tuple) else str(v)) for k, v in t["params"].items()}
        for m in t["history"]:
            rows.append({
                "trial_id": trial_id,
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "train_acc": m.train_acc,
                "val_acc": m.val_acc,
                "train_f1": m.train_f1,
                "val_f1": m.val_f1,
                **params
            })
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df
