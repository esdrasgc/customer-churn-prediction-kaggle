"""
Training module: MLP + Optuna with CV on training, validation-driven optimization,
and comprehensive evaluation and plots for train/val/test.

Usage (inside a notebook):

from src.train_mlp_cv_optuna import run_training
results = run_training(X, y, n_trials=20, random_state=42, cv_folds=4, verbose=True)

This will:
- Split data like your notebook (80% per class to train; remaining split equally into val/test)
- Run Optuna minimizing validation log loss while performing CV on the training set
- Train best model and report metrics + plots on train, val, and test
- Refit on train+val and report final test metrics + plots

Returned `results` contains:
{
  'study': optuna.study.Study,
  'best_params': dict,
  'splits': { 'train_idx', 'val_idx', 'test_idx' },
  'metrics': {
      'train': {...}, 'val': {...}, 'test': {...},
      'final_test_refit_trainval': {...}
  }
}

Note: Copy this file or its functions into your notebook as needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
)

warnings.filterwarnings("ignore")


# -----------------------------
# Data splitting utility
# -----------------------------

def stratified_custom_split_indices(y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced splits with class parity using a 70/15/15 rule.
    Each split (train/val/test) will have the SAME number of positives and negatives.
    We base counts on the minority class to ensure feasibility.

    Returns (train_idx, val_idx, test_idx)
    """
    y_flat = y.ravel()

    pos_idx = np.where(y_flat == 1)[0]
    neg_idx = np.where(y_flat == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_min = min(n_pos, n_neg)

    # Target per-class counts using 70/15/15 split. Work with integers.
    train_per_class = int(np.floor(0.70 * n_min))
    val_per_class = int(np.floor(0.15 * n_min))
    test_per_class = int(np.floor(0.15 * n_min))

    # Adjust for any rounding remainder by assigning to train (keeps 70% dominant)
    remainder = n_min - (train_per_class + val_per_class + test_per_class)
    train_per_class += remainder

    rng = np.random.RandomState(random_state)

    # Sample per class without replacement
    pos_perm = rng.permutation(pos_idx)
    neg_perm = rng.permutation(neg_idx)

    pos_train = pos_perm[:train_per_class]
    pos_val = pos_perm[train_per_class:train_per_class + val_per_class]
    pos_test = pos_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

    neg_train = neg_perm[:train_per_class]
    neg_val = neg_perm[train_per_class:train_per_class + val_per_class]
    neg_test = neg_perm[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]

    # Combine splits and shuffle indices within each split for randomness
    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


# -----------------------------
# Model/pipeline and objective
# -----------------------------

def build_pipeline(trial: optuna.Trial, random_state: int) -> Pipeline:
    """Create a StandardScaler + MLPClassifier pipeline with hyperparameters from Optuna."""
    hidden_layer_sizes = []
    n_layers = trial.suggest_int("n_layers", 1, 5)
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 50, 500)
        hidden_layer_sizes.append(layer_size)

    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])  # logistic = sigmoid
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])  # lbfgs often strong on smaller datasets
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True)

    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])
    return pipe


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    cv_folds: int,
) -> float:
    """Optuna objective that:
    - Builds a pipeline
    - Runs CV (accuracy) on training data for robustness signal
    - Fits on full training data
    - Computes validation log loss and returns it to be MINIMIZED
    """
    pipe = build_pipeline(trial, random_state)

    # CV on training set (does not see validation)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_cv_acc = float(np.mean(cv_scores))

    # Fit on full training, evaluate on validation
    pipe.fit(X_train, y_train)

    # Use probabilities for log loss and ROC/PR later
    y_val_proba = pipe.predict_proba(X_val)[:, 1]
    val_logloss = float(log_loss(y_val, y_val_proba))
    val_acc = float(accuracy_score(y_val, (y_val_proba >= 0.5).astype(int)))

    # Store extra info for analysis
    trial.set_user_attr("mean_cv_acc", mean_cv_acc)
    trial.set_user_attr("val_acc_from_best_fit", val_acc)

    return val_logloss  # MINIMIZE validation error (log loss)


# -----------------------------
# Evaluation helpers
# -----------------------------

@dataclass
class EvalResult:
    accuracy: float
    logloss: float
    report: str


def evaluate_split(pipe: Pipeline, X: np.ndarray, y: np.ndarray, title_prefix: str, plot: bool = True) -> EvalResult:
    y_proba = pipe.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    ll = log_loss(y, y_proba)
    rep = classification_report(y, y_pred, digits=4)

    if plot:
        plot_all(y_true=y, y_pred=y_pred, y_proba=y_proba, title_prefix=title_prefix)

    return EvalResult(accuracy=acc, logloss=ll, report=rep)


def plot_all(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{title_prefix} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title(f"{title_prefix} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[2].plot(recall, precision, label=f"AP = {ap:.4f}")
    axes[2].set_title(f"{title_prefix} - Precision-Recall")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Orchestrator
# -----------------------------

def run_training(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 2,
    random_state: int = 42,
    cv_folds: int = 4,
    verbose: bool = True,
    save_best_model_path: Optional[str] = None,
    save_study_path: Optional[str] = None,
    save_final_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full training procedure described.

    - Custom stratified split (80% per class -> train; rest -> val/test)
    - Optuna minimizing validation log loss; CV on training set
    - Evaluate best model on train, val, test
    - Refit on train+val and evaluate on test again (final)
    """
    # 1) Create splits
    train_idx, val_idx, test_idx = stratified_custom_split_indices(y, random_state)
    X_train, y_train = X[train_idx], y[train_idx].ravel()
    X_val, y_val = X[val_idx], y[val_idx].ravel()
    X_test, y_test = X[test_idx], y[test_idx].ravel()

    if verbose:
        from collections import Counter
        print("train shape:", X_train.shape, "class counts:", Counter(y_train))
        print("val   shape:", X_val.shape,   "class counts:", Counter(y_val))
        print("test  shape:", X_test.shape,  "class counts:", Counter(y_test))

    # 2) Optuna study: minimize validation log loss
    study = optuna.create_study(direction="minimize")

    if verbose:
        print(f"Iniciando otimização com {n_trials} trials... (direção: minimizar log loss de validação)")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, random_state, cv_folds),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    if verbose:
        print("Otimização concluída!")
        print(f"Melhor valor (val log loss): {study.best_value:.6f}")
        print(f"Melhores parâmetros: {study.best_params}")
    # Optional: persist the study
    if save_study_path is not None:
        with open(save_study_path, "wb") as f:
            pickle.dump(study, f)

    # 3) Train best model on TRAIN only, evaluate on train/val/test
    best_pipe = build_pipeline(study.best_trial, random_state)
    best_pipe.fit(X_train, y_train)
    # Optional: persist best model (trained on TRAIN only)
    if save_best_model_path is not None:
        with open(save_best_model_path, "wb") as f:
            pickle.dump(best_pipe, f)

    if verbose:
        print("\nAvaliação com melhor modelo (treinado apenas em TRAIN):")

    train_eval = evaluate_split(best_pipe, X_train, y_train, title_prefix="Train")
    if verbose:
        print("[TRAIN] accuracy=%.4f logloss=%.6f" % (train_eval.accuracy, train_eval.logloss))
        print(train_eval.report)

    val_eval = evaluate_split(best_pipe, X_val, y_val, title_prefix="Validação")
    if verbose:
        print("[VAL]   accuracy=%.4f logloss=%.6f" % (val_eval.accuracy, val_eval.logloss))
        print(val_eval.report)

    test_eval = evaluate_split(best_pipe, X_test, y_test, title_prefix="Teste (modelo treinado em TRAIN)")
    if verbose:
        print("[TEST]  accuracy=%.4f logloss=%.6f" % (test_eval.accuracy, test_eval.logloss))
        print(test_eval.report)

    # 4) Refit on TRAIN+VAL and evaluate on TEST (final report)
    X_trval = np.vstack([X_train, X_val])
    y_trval = np.concatenate([y_train, y_val])

    final_pipe = build_pipeline(study.best_trial, random_state)
    final_pipe.fit(X_trval, y_trval)
    # Optional: persist final refit model (TRAIN+VAL)
    if save_final_model_path is not None:
        with open(save_final_model_path, "wb") as f:
            pickle.dump(final_pipe, f)

    if verbose:
        print("\nAvaliação FINAL no TESTE (modelo treinado em TRAIN+VAL):")

    final_test_eval = evaluate_split(final_pipe, X_test, y_test, title_prefix="Teste FINAL (modelo treinado em TRAIN+VAL)")
    if verbose:
        print("[FINAL TEST] accuracy=%.4f logloss=%.6f" % (final_test_eval.accuracy, final_test_eval.logloss))
        print(final_test_eval.report)

    return {
        "study": study,
        "best_params": study.best_params,
        "splits": {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        "metrics": {
            "train": {
                "accuracy": train_eval.accuracy,
                "logloss": train_eval.logloss,
                "report": train_eval.report,
            },
            "val": {
                "accuracy": val_eval.accuracy,
                "logloss": val_eval.logloss,
                "report": val_eval.report,
            },
            "test": {
                "accuracy": test_eval.accuracy,
                "logloss": test_eval.logloss,
                "report": test_eval.report,
            },
            "final_test_refit_trainval": {
                "accuracy": final_test_eval.accuracy,
                "logloss": final_test_eval.logloss,
                "report": final_test_eval.report,
            },
        },
    }


if __name__ == "__main__":
    results = run_training(
        X, y,
        n_trials=20,
        random_state=42,
        cv_folds=4,
        verbose=True,
        # Optional persistence (uncomment to save):
        save_best_model_path="artifacts/best_model_train.pkl",
        save_final_model_path="artifacts/final_model_trainval.pkl",
        save_study_path="artifacts/optuna_study.pkl",
    )