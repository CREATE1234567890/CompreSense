import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def _compute_binary_metrics(y_true_np: np.ndarray, y_score_np: np.ndarray, threshold: float) -> dict:
    y_pred_np = (y_score_np >= threshold).astype(np.int32)

    unique_classes = np.unique(y_true_np)
    auc = float(roc_auc_score(y_true_np, y_score_np)) if len(unique_classes) >= 2 else float("nan")
    acc = float(accuracy_score(y_true_np, y_pred_np))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "acc": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def _search_best_threshold(y_true_np: np.ndarray, y_score_np: np.ndarray, optimize_for: str) -> float:
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score_np >= t).astype(np.int32)
        if optimize_for == "f1":
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_np,
                y_pred,
                average="binary",
                zero_division=0,
            )
            score = float(f1)
        else:
            score = float(accuracy_score(y_true_np, y_pred))

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def evaluate_prediction_file(
    pred_path: str | Path,
    gold_path: str | Path,
    threshold: float = 0.5,
    tune_threshold_for: str = "none",
) -> dict:
    """Evaluate prediction JSONL against gold JSONL and return common binary metrics."""
    pred_path = Path(pred_path)
    gold_path = Path(gold_path)

    gold = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            gold[j["id"]] = int(j["label"])

    y_true = []
    y_score = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            doc_id = j["id"]
            if doc_id in gold:
                y_true.append(gold[doc_id])
                y_score.append(float(j["label"]))

    if not y_true:
        raise ValueError("No overlapping IDs between prediction and gold files")

    y_true_np = np.array(y_true, dtype=np.int32)
    y_score_np = np.array(y_score, dtype=np.float32)
    threshold_used = float(threshold)
    if tune_threshold_for in {"acc", "f1"}:
        threshold_used = _search_best_threshold(y_true_np, y_score_np, optimize_for=tune_threshold_for)

    metrics = _compute_binary_metrics(y_true_np, y_score_np, threshold_used)
    metrics["n"] = int(len(y_true_np))
    metrics["threshold_mode"] = tune_threshold_for
    return metrics