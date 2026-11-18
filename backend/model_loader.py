"""Model loading utilities with version checking and validation."""
import json
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import joblib
import sklearn
import numpy as np

from config import (
    STAGE1_MODEL_PATH, STAGE1_SCALER_PATH, STAGE1_PCA_PATH,
    STAGE1_FEATURES_PATH, STAGE1_LABELS_PATH,
    STAGE2_MODEL_PATH, STAGE2_SCALER_PATH,
    STAGE2_FEATURES_PATH, STAGE2_LABELS_PATH,
    METADATA_PATH, STAGE1_PCA_COMPONENTS, STAGE1_EXPECTED_VARIANCE
)


class ModelLoadError(Exception):
    """Raised when model artifacts cannot be loaded or validated."""
    pass


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_sklearn_version(metadata: Dict) -> Tuple[bool, str]:
    """Check if current scikit-learn version matches training version."""
    trained_version = metadata.get("sklearn_version", "unknown")
    current_version = sklearn.__version__
    match = trained_version == current_version
    msg = f"scikit-learn version: trained={trained_version}, current={current_version}"
    return match, msg


def validate_feature_names(scaler, expected_features: list, stage_name: str) -> Tuple[bool, str]:
    """Validate that scaler's feature_names_in_ matches expected features."""
    if not hasattr(scaler, "feature_names_in_"):
        return True, f"{stage_name}: scikit-learn < 1.0, feature_names_in_ not available"
    
    actual_features = list(scaler.feature_names_in_)
    if actual_features != expected_features:
        return False, (
            f"{stage_name}: Feature mismatch!\n"
            f"  Expected ({len(expected_features)}): {expected_features[:5]}...\n"
            f"  Actual ({len(actual_features)}): {actual_features[:5]}..."
        )
    return True, f"{stage_name}: Feature names match ({len(expected_features)} features)"


def validate_pca(pca, expected_components: int, expected_variance: float) -> Tuple[bool, str]:
    """Validate PCA components and explained variance."""
    if pca.n_components_ != expected_components:
        return False, f"PCA: Expected {expected_components} components, got {pca.n_components_}"
    
    if hasattr(pca, "explained_variance_ratio_"):
        actual_variance = float(pca.explained_variance_ratio_.sum())
        if abs(actual_variance - expected_variance) > 0.01:
            return False, f"PCA: Expected variance ~{expected_variance:.4f}, got {actual_variance:.4f}"
    
    return True, f"PCA: {expected_components} components, variance ~{expected_variance:.4f}"


def load_stage1_artifacts() -> Dict[str, Any]:
    """Load all Stage 1 artifacts with validation."""
    artifacts = {}
    errors = []
    warnings_list = []
    
    # Load metadata
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    
    # Check scikit-learn version
    version_match, version_msg = check_sklearn_version(metadata)
    if not version_match:
        warnings_list.append(f"Version mismatch: {version_msg}")
    
    # Load model
    try:
        artifacts["model"] = joblib.load(STAGE1_MODEL_PATH)
    except Exception as e:
        errors.append(f"Failed to load Stage 1 model: {e}")
    
    # Load scaler
    try:
        artifacts["scaler"] = joblib.load(STAGE1_SCALER_PATH)
    except Exception as e:
        errors.append(f"Failed to load Stage 1 scaler: {e}")
    
    # Load PCA
    try:
        artifacts["pca"] = joblib.load(STAGE1_PCA_PATH)
    except Exception as e:
        errors.append(f"Failed to load Stage 1 PCA: {e}")
    
    # Load feature columns
    try:
        with open(STAGE1_FEATURES_PATH, "r") as f:
            artifacts["feature_columns"] = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load Stage 1 feature columns: {e}")
    
    # Load label mapping
    try:
        with open(STAGE1_LABELS_PATH, "r") as f:
            artifacts["label_mapping"] = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load Stage 1 label mapping: {e}")
    
    # Validate
    if "scaler" in artifacts and "feature_columns" in artifacts:
        valid, msg = validate_feature_names(
            artifacts["scaler"], artifacts["feature_columns"], "Stage 1"
        )
        if not valid:
            errors.append(msg)
        else:
            warnings_list.append(msg)
    
    if "pca" in artifacts:
        valid, msg = validate_pca(artifacts["pca"], STAGE1_PCA_COMPONENTS, STAGE1_EXPECTED_VARIANCE)
        if not valid:
            errors.append(msg)
        else:
            warnings_list.append(msg)
    
    # Verify hashes if metadata available
    if metadata.get("artifacts"):
        for name, expected_hash in metadata["artifacts"].items():
            path_map = {
                "stage1_model": STAGE1_MODEL_PATH,
                "stage1_scaler": STAGE1_SCALER_PATH,
                "stage1_pca": STAGE1_PCA_PATH,
            }
            if name in path_map and path_map[name].exists():
                actual_hash = compute_file_hash(path_map[name])
                if actual_hash != expected_hash:
                    warnings_list.append(f"Hash mismatch for {name}: expected {expected_hash[:8]}, got {actual_hash[:8]}")
    
    artifacts["metadata"] = metadata
    artifacts["errors"] = errors
    artifacts["warnings"] = warnings_list
    
    if errors:
        raise ModelLoadError(f"Stage 1 load errors: {'; '.join(errors)}")
    
    return artifacts


def load_stage2_artifacts() -> Dict[str, Any]:
    """Load all Stage 2 artifacts with validation."""
    artifacts = {}
    errors = []
    warnings_list = []
    
    # Load metadata
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    
    # Check scikit-learn version
    version_match, version_msg = check_sklearn_version(metadata)
    if not version_match:
        warnings_list.append(f"Version mismatch: {version_msg}")
    
    # Load model
    try:
        artifacts["model"] = joblib.load(STAGE2_MODEL_PATH)
    except Exception as e:
        errors.append(f"Failed to load Stage 2 model: {e}")
    
    # Load scaler
    try:
        artifacts["scaler"] = joblib.load(STAGE2_SCALER_PATH)
    except Exception as e:
        errors.append(f"Failed to load Stage 2 scaler: {e}")
    
    # Load feature columns
    try:
        with open(STAGE2_FEATURES_PATH, "r") as f:
            artifacts["feature_columns"] = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load Stage 2 feature columns: {e}")
    
    # Load label mapping
    try:
        with open(STAGE2_LABELS_PATH, "r") as f:
            artifacts["label_mapping"] = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load Stage 2 label mapping: {e}")
    
    # Validate
    if "scaler" in artifacts and "feature_columns" in artifacts:
        valid, msg = validate_feature_names(
            artifacts["scaler"], artifacts["feature_columns"], "Stage 2"
        )
        if not valid:
            errors.append(msg)
        else:
            warnings_list.append(msg)
    
    # Verify hashes
    if metadata.get("artifacts"):
        for name, expected_hash in metadata["artifacts"].items():
            path_map = {
                "stage2_model": STAGE2_MODEL_PATH,
                "stage2_scaler": STAGE2_SCALER_PATH,
            }
            if name in path_map and path_map[name].exists():
                actual_hash = compute_file_hash(path_map[name])
                if actual_hash != expected_hash:
                    warnings_list.append(f"Hash mismatch for {name}: expected {expected_hash[:8]}, got {actual_hash[:8]}")
    
    artifacts["metadata"] = metadata
    artifacts["errors"] = errors
    artifacts["warnings"] = warnings_list
    
    if errors:
        raise ModelLoadError(f"Stage 2 load errors: {'; '.join(errors)}")
    
    return artifacts


def load_all_artifacts() -> Tuple[Dict, Dict]:
    """Load both Stage 1 and Stage 2 artifacts."""
    stage1 = load_stage1_artifacts()
    stage2 = load_stage2_artifacts()
    return stage1, stage2


def save_metadata(metadata: Dict) -> None:
    """Save metadata to JSON file."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)