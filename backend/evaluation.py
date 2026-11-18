"""Evaluation metrics computation."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_binary_metrics(
    y_true: List[str],
    y_pred: List[str],
    benign_label: str = "BENIGN"
) -> Dict:
    """Compute binary classification metrics (BENIGN vs ATTACK)."""
    y_true_bin = ["BENIGN" if str(y).upper() == "BENIGN" else "ATTACK" for y in y_true]
    y_pred_bin = ["BENIGN" if str(y).upper() == "BENIGN" else "ATTACK" for y in y_pred]
    
    labels = ["BENIGN", "ATTACK"]
    
    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(
            y_true_bin, y_pred_bin, average="binary", pos_label="ATTACK", zero_division=0
        ),
        "recall": recall_score(
            y_true_bin, y_pred_bin, average="binary", pos_label="ATTACK", zero_division=0
        ),
        "f1": f1_score(
            y_true_bin, y_pred_bin, average="binary", pos_label="ATTACK", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin, labels=labels).tolist(),
        "labels": labels,
        "support": {
            "BENIGN": y_true_bin.count("BENIGN"),
            "ATTACK": y_true_bin.count("ATTACK"),
        },
    }


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict:
    """Compute multiclass classification metrics for attack categories."""
    # Filter to only attack rows
    attack_mask = [str(y).upper() != "BENIGN" for y in y_true]
    
    if not any(attack_mask):
        return {
            "accuracy": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": [],
            "labels": [],
            "per_class": {},
            "support": {},
            "warning": "No attack samples in ground truth",
        }
    
    y_true_att = [y_true[i] for i in range(len(y_true)) if attack_mask[i]]
    y_pred_att = [y_pred[i] for i in range(len(y_pred)) if attack_mask[i]]
    
    if labels is None:
        labels = sorted(set(y_true_att) | set(y_pred_att))
    
    cm = confusion_matrix(y_true_att, y_pred_att, labels=labels)
    
    per_class = {}
    for i, label in enumerate(labels):
        support = int(cm[i].sum())
        if support > 0:
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    
    return {
        "accuracy": accuracy_score(y_true_att, y_pred_att),
        "precision_weighted": precision_score(y_true_att, y_pred_att, average="weighted", zero_division=0, labels=labels),
        "recall_weighted": recall_score(y_true_att, y_pred_att, average="weighted", zero_division=0, labels=labels),
        "f1_weighted": f1_score(y_true_att, y_pred_att, average="weighted", zero_division=0, labels=labels),
        "precision_macro": precision_score(y_true_att, y_pred_att, average="macro", zero_division=0, labels=labels),
        "recall_macro": recall_score(y_true_att, y_pred_att, average="macro", zero_division=0, labels=labels),
        "f1_macro": f1_score(y_true_att, y_pred_att, average="macro", zero_division=0, labels=labels),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
        "support": {label: per_class[label]["support"] for label in labels},
    }


def compute_session_metrics(results: List[Dict]) -> Dict:
    """Compute all metrics from session results."""
    # Extract predictions and ground truth
    y_pred_final = [r["final_label"] for r in results]
    y_pred_stage1 = [r["stage1_label"] for r in results]
    y_true = [r["ground_truth"] for r in results if r.get("ground_truth") is not None]
    
    metrics = {
        "total_flows": len(results),
        "flows_with_ground_truth": len(y_true),
    }
    
    # Overall distribution
    pred_dist = pd.Series(y_pred_final).value_counts().to_dict()
    metrics["prediction_distribution"] = pred_dist
    metrics["benign_count"] = pred_dist.get("BENIGN", 0)
    metrics["attack_count"] = sum(v for k, v in pred_dist.items() if k != "BENIGN")
    
    if y_true:
        y_true_list = y_true
        # Align predictions with ground truth
        y_pred_aligned = [r["final_label"] for r in results if r.get("ground_truth") is not None]
        y_pred_stage1_aligned = [r["stage1_label"] for r in results if r.get("ground_truth") is not None]
        
        # Stage 1 (binary) metrics
        metrics["stage1"] = compute_binary_metrics(y_true_list, y_pred_stage1_aligned)
        
        # Stage 2 (multiclass) metrics - only flows that reached Stage 2
        # (Stage 1 predicted ATTACK). Stage 2 never outputs BENIGN, so missed
        # detections (Stage 1 = BENIGN, ground truth = attack) are excluded
        # here; they are already captured by Stage 1 recall / false negatives.
        s2_mask = [p == "ATTACK" for p in y_pred_stage1_aligned]
        y_true_s2 = [y for y, m in zip(y_true_list, s2_mask) if m]
        y_pred_s2 = [p for p, m in zip(y_pred_aligned, s2_mask) if m]
        metrics["stage2"] = compute_multiclass_metrics(y_true_s2, y_pred_s2)
        
        # Overall metrics
        metrics["overall"] = {
            "accuracy": accuracy_score(y_true_list, y_pred_aligned),
            "correct_predictions": sum(1 for r in results if r.get("correct") is True),
            "incorrect_predictions": sum(1 for r in results if r.get("correct") is False),
            "false_positives": sum(
                1 for r in results 
                if r.get("ground_truth", "").upper() == "BENIGN" and r.get("final_label") != "BENIGN"
            ),
            "false_negatives": sum(
                1 for r in results 
                if r.get("ground_truth", "").upper() != "BENIGN" and r.get("final_label") == "BENIGN"
            ),
        }
    
    return metrics


def format_confusion_matrix(cm: List[List[int]], labels: List[str]) -> pd.DataFrame:
    """Format confusion matrix as DataFrame for display."""
    return pd.DataFrame(cm, index=labels, columns=labels)


def format_per_class_table(per_class: Dict) -> pd.DataFrame:
    """Format per-class metrics as DataFrame."""
    rows = []
    for label, metrics in per_class.items():
        rows.append({
            "Class": label,
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "Support": metrics['support'],
        })
    return pd.DataFrame(rows)