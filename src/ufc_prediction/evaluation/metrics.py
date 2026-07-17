"""Reusable probability and classification metrics for UFC model evaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def american_odds_to_prob(odds: Any) -> np.ndarray:
    """Convert American odds to their implied probability."""

    odds_array = np.asarray(odds, dtype=float)
    return np.where(
        odds_array < 0,
        -odds_array / (-odds_array + 100.0),
        100.0 / (odds_array + 100.0),
    )


def odds_implied_red_prob(red_odds: Any, blue_odds: Any) -> np.ndarray:
    """Return Red's normalized implied probability after removing bookmaker vig."""

    red_probability = american_odds_to_prob(red_odds)
    blue_probability = american_odds_to_prob(blue_odds)
    return red_probability / (red_probability + blue_probability)


def red_probability(model: Any, features: Any, red_label: Any) -> np.ndarray:
    """Return the probability column corresponding to a Red-corner win."""

    column = list(model.classes_).index(red_label)
    return model.predict_proba(features)[:, column]


def expected_calibration_error(
    y_true: Any,
    predicted_probability: Any,
    n_bins: int = 10,
) -> float:
    """Calculate expected calibration error using equal-width probability bins."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predicted_probability, dtype=float)
    bin_ids = np.digitize(predicted, np.linspace(0, 1, n_bins + 1)[1:-1])
    error = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if in_bin.any():
            error += in_bin.mean() * abs(actual[in_bin].mean() - predicted[in_bin].mean())
    return float(error)


def evaluate_binary(name: str, y_true_red: Any, p_red: Any) -> dict[str, Any]:
    """Evaluate binary winner probabilities with discrimination and calibration metrics."""

    actual = np.asarray(y_true_red)
    predicted = np.asarray(p_red, dtype=float)
    predicted_class = (predicted >= 0.5).astype(int)
    return {
        "Model": name,
        "n": len(actual),
        "Accuracy": accuracy_score(actual, predicted_class),
        "Log Loss": log_loss(actual, np.column_stack([1 - predicted, predicted]), labels=[0, 1]),
        "Brier": brier_score_loss(actual, predicted),
        "ROC-AUC": roc_auc_score(actual, predicted) if len(np.unique(actual)) > 1 else np.nan,
        "ECE": expected_calibration_error(actual, predicted),
    }


def evaluate_multiclass(name: str, model: Any, features: Any, labels: Any) -> dict[str, Any]:
    """Evaluate a multiclass model after excluding labels unseen during training."""

    labels_array = np.asarray(labels)
    known = np.isin(labels_array, model.classes_)
    if not known.all():
        print(f"{name}: dropped {(~known).sum()} test rows with classes unseen in training")
    known_features = features[known]
    known_labels = labels_array[known]
    probability = model.predict_proba(known_features)
    prediction = model.classes_[np.argmax(probability, axis=1)]
    return {
        "Model": name,
        "n": len(known_labels),
        "Accuracy": accuracy_score(known_labels, prediction),
        "Macro F1": f1_score(known_labels, prediction, average="macro", zero_division=0),
        "Weighted F1": f1_score(known_labels, prediction, average="weighted", zero_division=0),
        "Log Loss": log_loss(known_labels, probability, labels=model.classes_),
    }


def multiclass_metrics(name: str, model: Any, features: Any, labels: Any) -> dict[str, Any]:
    """Return the compact multiclass metric set used by method and round tasks."""

    probability = model.predict_proba(features)
    prediction = model.classes_[np.argmax(probability, axis=1)]
    return {
        "Model": name,
        "n": len(labels),
        "Accuracy": accuracy_score(labels, prediction),
        "Macro F1": f1_score(labels, prediction, average="macro", zero_division=0),
        "Log Loss": log_loss(labels, probability, labels=model.classes_),
    }
