"""Data preprocessing pipeline matching training exactly."""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    LEAKAGE_COLUMNS, INF_REPLACEMENT, NAN_REPLACEMENT,
    STAGE1_FEATURES_PATH, STAGE2_FEATURES_PATH
)


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label strings: strip, replace special chars."""
    labels = series.astype(str).str.strip()
    # Replace non-ASCII dash (–) and other special chars
    labels = labels.str.replace("–", "-", regex=False)  # en-dash
    labels = labels.str.replace("—", "-", regex=False)  # em-dash
    labels = labels.str.replace("'", "'", regex=False)  # right single quote
    labels = labels.str.replace("'", "'", regex=False)  # left single quote
    return labels


def handle_duplicate_columns(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Handle duplicate column names (e.g., 'Fwd Header Length.1').
    Training code typically keeps first occurrence; we match expected features.
    """
    df = df.copy()
    
    # Check for duplicates
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        # Keep first occurrence of each column name
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    
    return df


def select_features(df: pd.DataFrame, feature_columns: List[str], stage_name: str) -> pd.DataFrame:
    """Select and order features exactly as in training."""
    df = df.copy()
    
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise PreprocessingError(
            f"{stage_name}: Missing feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    
    # Select in exact order
    return df[feature_columns]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, replacing errors with NaN."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_inf_nan(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Replace inf with NaN, then fill NaN with 0. Return counts."""
    df = df.copy()
    counts = {"inf_replaced": 0, "nan_filled": 0}
    
    # Count and replace inf
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    counts["inf_replaced"] = int(inf_mask.sum().sum())
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count and fill NaN
    nan_mask = df.isna()
    counts["nan_filled"] = int(nan_mask.sum().sum())
    df = df.fillna(NAN_REPLACEMENT)
    
    return df, counts


def exclude_leakage_columns(df: pd.DataFrame, extra_leakage: List[str] = None) -> pd.DataFrame:
    """Drop leakage columns if present."""
    df = df.copy()
    to_drop = LEAKAGE_COLUMNS.copy()
    if extra_leakage:
        to_drop.extend(extra_leakage)
    
    existing = [c for c in to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def keep_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numeric columns."""
    return df.select_dtypes(include=[np.number])


def align_features(df: pd.DataFrame, expected_features: List[str], stage_name: str) -> pd.DataFrame:
    """Ensure DataFrame has exactly the expected features in order."""
    df = df.copy()
    
    # Add missing columns with 0
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
            warnings.warn(f"{stage_name}: Added missing feature '{feat}' with zeros")
    
    # Reorder and select
    return df[expected_features]


def load_feature_columns(stage: int) -> List[str]:
    """Load feature columns from JSON."""
    path = STAGE1_FEATURES_PATH if stage == 1 else STAGE2_FEATURES_PATH
    with open(path, "r") as f:
        return json.load(f)


def load_label_mapping(stage: int) -> Dict:
    """Load label mapping from JSON."""
    path = STAGE1_FEATURES_PATH.parent / "label_mapping.json" if stage == 1 \
        else STAGE2_FEATURES_PATH.parent / "label_mapping.json"
    with open(path, "r") as f:
        return json.load(f)


def preprocess_stage1(
    df: pd.DataFrame,
    feature_columns: List[str],
    is_training: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    Preprocess for Stage 1 (binary classification).
    Returns (X, y, preprocessing_info).
    """
    info = {"inf_nan_counts": {}, "dropped_columns": [], "shape_before": df.shape}
    
    df = normalize_columns(df)
    info["shape_after_normalize"] = df.shape
    
    # Extract labels if present
    y = None
    if "Label" in df.columns:
        y = normalize_labels(df["Label"])
        # Convert to binary: BENIGN vs ATTACK
        y_binary = y.apply(lambda x: "BENIGN" if str(x).upper() == "BENIGN" else "ATTACK")
        info["label_distribution"] = y_binary.value_counts().to_dict()
    
    # Drop leakage columns
    df = exclude_leakage_columns(df)
    info["dropped_columns"] = LEAKAGE_COLUMNS
    
    # Handle duplicates
    df = handle_duplicate_columns(df, feature_columns)
    
    # Select features
    X = select_features(df, feature_columns, "Stage 1")
    info["shape_after_select"] = X.shape
    
    # Coerce numeric
    X = coerce_numeric(X)
    
    # Replace inf/nan
    X, counts = replace_inf_nan(X)
    info["inf_nan_counts"] = counts
    
    return X, y, info


def preprocess_stage2(
    df: pd.DataFrame,
    feature_columns: List[str],
    is_training: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    Preprocess for Stage 2 (attack classification).
    Only processes rows where Label != BENIGN.
    Returns (X, y, preprocessing_info).
    """
    info = {"inf_nan_counts": {}, "dropped_columns": [], "shape_before": df.shape}
    
    df = normalize_columns(df)
    info["shape_after_normalize"] = df.shape
    
    # Extract labels if present
    y = None
    if "Label" in df.columns:
        y = normalize_labels(df["Label"])
        info["label_distribution"] = y.value_counts().to_dict()
    
    # Drop leakage columns (including Attack if present)
    extra_leakage = ["Attack"] if "Attack" in df.columns else []
    df = exclude_leakage_columns(df, extra_leakage)
    info["dropped_columns"] = LEAKAGE_COLUMNS + extra_leakage
    
    # Handle duplicates
    df = handle_duplicate_columns(df, feature_columns)
    
    # Select features
    X = select_features(df, feature_columns, "Stage 2")
    info["shape_after_select"] = X.shape
    
    # Coerce numeric
    X = coerce_numeric(X)
    
    # Replace inf/nan
    X, counts = replace_inf_nan(X)
    info["inf_nan_counts"] = counts
    
    return X, y, info


def preprocess_for_inference(
    df: pd.DataFrame,
    stage1_features: List[str],
    stage2_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Preprocess a single DataFrame for both stages.
    Returns (X_stage1, X_stage2, info).
    """
    info = {}
    
    df = normalize_columns(df)
    
    # Stage 1 preprocessing
    X1 = df.copy()
    X1 = exclude_leakage_columns(X1)
    X1 = handle_duplicate_columns(X1, stage1_features)
    X1 = align_features(X1, stage1_features, "Stage 1")
    X1 = coerce_numeric(X1)
    X1, counts1 = replace_inf_nan(X1)
    info["stage1_inf_nan"] = counts1
    
    # Stage 2 preprocessing
    X2 = df.copy()
    extra_leakage = ["Attack"] if "Attack" in X2.columns else []
    X2 = exclude_leakage_columns(X2, extra_leakage)
    X2 = handle_duplicate_columns(X2, stage2_features)
    X2 = align_features(X2, stage2_features, "Stage 2")
    X2 = coerce_numeric(X2)
    X2, counts2 = replace_inf_nan(X2)
    info["stage2_inf_nan"] = counts2
    
    return X1, X2, info