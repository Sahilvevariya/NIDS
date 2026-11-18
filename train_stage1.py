"""Stage 1: Binary Classification Training (BENIGN vs ATTACK) with PCA."""
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from config import (
    STAGE1_DIR, STAGE1_MODEL_PATH, STAGE1_SCALER_PATH, STAGE1_PCA_PATH,
    STAGE1_FEATURES_PATH, STAGE1_LABELS_PATH, METADATA_PATH,
    STAGE1_PCA_COMPONENTS, LEAKAGE_COLUMNS, STAGE2_EXTRA_LEAKAGE,
    INF_REPLACEMENT, NAN_REPLACEMENT,
)


def compute_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_prepare_data(data_dir: Path) -> pd.DataFrame:
    """Load all CSV files and prepare for Stage 1 training."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for f in csv_files:
        print(f"Loading {f.name}...")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined)}")
    return combined


def preprocess_stage1_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Preprocess for Stage 1 training - returns X, y, feature_columns."""
    df = df.copy()
    
    # Normalize labels
    df["Label"] = df["Label"].astype(str).str.strip()
    
    # Create binary label
    y = df["Label"].apply(lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK")
    
    # Drop leakage columns
    to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=to_drop)
    
    # Handle duplicate columns
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    
    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    
    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(NAN_REPLACEMENT)
    
    feature_columns = list(X.columns)
    print(f"Stage 1 features: {len(feature_columns)}")
    
    return X, y, feature_columns


def train_stage1(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    pca_components: int = 25,
) -> Dict:
    """Train Stage 1 model and export artifacts."""
    print("=" * 60)
    print("STAGE 1: BINARY CLASSIFICATION TRAINING")
    print("=" * 60)
    
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data(data_dir)
    
    # Preprocess
    X, y, feature_columns = preprocess_stage1_training(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Train distribution:\n{y_train.value_counts()}")
    print(f"Test distribution:\n{y_test.value_counts()}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.4f}")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    
    print("Training Random Forest...")
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_acc:.6f}")
    print(f"Testing Accuracy: {test_acc:.6f}")
    print(f"Difference: {train_acc - test_acc:.6f}")
    
    print("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_test_pred, labels=["BENIGN", "ATTACK"])
    print(cm)
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    # Label mapping
    label_mapping = {"BENIGN": 0, "ATTACK": 1}
    
    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, STAGE1_MODEL_PATH)
    joblib.dump(scaler, STAGE1_SCALER_PATH)
    joblib.dump(pca, STAGE1_PCA_PATH)
    
    with open(STAGE1_FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    
    with open(STAGE1_LABELS_PATH, "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save metadata
    metadata = {
        "stage": 1,
        "sklearn_version": joblib.__version__,
        "random_state": random_state,
        "test_size": test_size,
        "pca_components": pca_components,
        "explained_variance": float(explained_var),
        "n_features": len(feature_columns),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_distribution": y_train.value_counts().to_dict(),
        "test_distribution": y_test.value_counts().to_dict(),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "artifacts": {
            "stage1_model": compute_hash(STAGE1_MODEL_PATH),
            "stage1_scaler": compute_hash(STAGE1_SCALER_PATH),
            "stage1_pca": compute_hash(STAGE1_PCA_PATH),
        },
    }
    
    import json as json_lib
    with open(METADATA_PATH, "w") as f:
        json_lib.dump(metadata, f, indent=2)
    
    print(f"Artifacts saved to {STAGE1_DIR}")
    
    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
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
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    train_stage1(data_dir)