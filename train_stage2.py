"""Stage 2: Multiclass Attack Classification Training."""
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    STAGE2_DIR, STAGE2_MODEL_PATH, STAGE2_SCALER_PATH,
    STAGE2_FEATURES_PATH, STAGE2_LABELS_PATH, METADATA_PATH,
    STAGE2_MAX_SAMPLES_PER_CLASS, STAGE2_RANDOM_STATE, STAGE2_TEST_SIZE,
    LEAKAGE_COLUMNS, INF_REPLACEMENT, NAN_REPLACEMENT,
)


def compute_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_prepare_data(data_dir: Path) -> pd.DataFrame:
    """Load all CSV files and prepare for Stage 2 training (attack only)."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    attack_data = []
    for f in csv_files:
        print(f"Loading {f.name}...")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
            # Remove BENIGN
            df = df[df["Label"].str.upper() != "BENIGN"]
        
        attack_data.append(df)
    
    attack_df = pd.concat(attack_data, ignore_index=True)
    print(f"Total attack records: {len(attack_df)}")
    return attack_df


def preprocess_stage2_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Preprocess for Stage 2 training."""
    df = df.copy()
    
    # Normalize labels
    df["Label"] = df["Label"].astype(str).str.strip()
    y = df["Label"].copy()
    
    # Drop leakage columns
    remove_columns = LEAKAGE_COLUMNS.copy()
    if "Attack" in df.columns:
        remove_columns.append("Attack")
    
    features = [c for c in df.columns if c not in remove_columns]
    X = df[features].copy()
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(NAN_REPLACEMENT)
    
    # Keep numeric only
    X = X.select_dtypes(include=[np.number])
    
    # Handle duplicates
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    
    # Ensure y aligned
    assert len(X) == len(y), "X and y length mismatch after preprocessing"
    
    feature_columns = list(X.columns)
    print(f"Stage 2 features: {len(feature_columns)}")
    
    return X, y, feature_columns


def controlled_sampling(X: pd.DataFrame, y: pd.Series, max_per_class: int, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Sample up to max_per_class per attack category."""
    balanced_parts = []
    
    for label, group_indices in y.groupby(y).groups.items():
        group_X = X.loc[list(group_indices)]
        group_y = y.loc[list(group_indices)]
        
        n = min(len(group_X), max_per_class)
        sampled_idx = group_X.sample(n=n, random_state=random_state).index
        
        balanced_parts.append((group_X.loc[sampled_idx], group_y.loc[sampled_idx]))
    
    X_balanced = pd.concat([p[0] for p in balanced_parts], ignore_index=True)
    y_balanced = pd.concat([p[1] for p in balanced_parts], ignore_index=True)
    
    print(f"Balanced dataset: {len(X_balanced)} samples")
    print(y_balanced.value_counts())
    
    return X_balanced, y_balanced


def train_stage2(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples_per_class: int = 10000,
) -> Dict:
    """Train Stage 2 model and export artifacts."""
    print("=" * 60)
    print("STAGE 2: MULTICLASS ATTACK CLASSIFICATION TRAINING")
    print("=" * 60)
    
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data(data_dir)
    
    # Preprocess
    X, y, feature_columns = preprocess_stage2_training(df)
    
    # Controlled sampling
    X_balanced, y_balanced = controlled_sampling(X, y, max_samples_per_class, random_state)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state, stratify=y_balanced
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"\nTraining distribution:\n{y_train.value_counts()}")
    print(f"\nTesting distribution:\n{y_test.value_counts()}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    
    print("Training Random Forest...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining Accuracy: {train_acc:.6f}")
    print(f"Testing Accuracy: {test_acc:.6f}")
    print(f"Difference: {train_acc - test_acc:.6f}")
    
    print(f"\nWeighted Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.6f}")
    print(f"Weighted Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.6f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.6f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print(cm_df)
    
    # Label mapping from model classes
    label_mapping = {cls: idx for idx, cls in enumerate(model.classes_)}
    
    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, STAGE2_MODEL_PATH)
    joblib.dump(scaler, STAGE2_SCALER_PATH)
    
    with open(STAGE2_FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    
    with open(STAGE2_LABELS_PATH, "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Update metadata
    import json as json_lib
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json_lib.load(f)
    
    metadata["stage2"] = {
        "sklearn_version": joblib.__version__,
        "random_state": random_state,
        "test_size": test_size,
        "max_samples_per_class": max_samples_per_class,
        "n_features": len(feature_columns),
        "n_classes": len(model.classes_),
        "classes": list(model.classes_),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_distribution": y_train.value_counts().to_dict(),
        "test_distribution": y_test.value_counts().to_dict(),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "weighted_precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(model.classes_),
        "artifacts": {
            "stage2_model": compute_hash(STAGE2_MODEL_PATH),
            "stage2_scaler": compute_hash(STAGE2_SCALER_PATH),
        },
    }
    
    with open(METADATA_PATH, "w") as f:
        json_lib.dump(metadata, f, indent=2)
    
    print(f"Artifacts saved to {STAGE2_DIR}")
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "label_mapping": label_mapping,
        "metrics": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "confusion_matrix": cm.tolist(),
        },
    }


if __name__ == "__main__":
    import sys
    from typing import Tuple
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    train_stage2(data_dir)