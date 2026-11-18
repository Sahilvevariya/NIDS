"""Inference pipeline for two-stage intrusion detection."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from config import LEAKAGE_COLUMNS
from backend.preprocessing import (
    preprocess_for_inference,
    load_feature_columns,
    load_label_mapping,
    normalize_labels,
)
from backend.model_loader import load_stage1_artifacts, load_stage2_artifacts


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class TwoStageDetector:
    """Two-stage intrusion detection pipeline."""
    
    def __init__(self):
        self.stage1_artifacts = None
        self.stage2_artifacts = None
        self.stage1_features = None
        self.stage2_features = None
        self.stage1_label_map = None
        self.stage2_label_map = None
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        """Load all model artifacts."""
        if self._loaded:
            return {"status": "already_loaded"}
        
        # Load artifacts
        self.stage1_artifacts = load_stage1_artifacts()
        self.stage2_artifacts = load_stage2_artifacts()
        
        # Extract components
        self.stage1_features = self.stage1_artifacts["feature_columns"]
        self.stage2_features = self.stage2_artifacts["feature_columns"]
        self.stage1_label_map = self.stage1_artifacts["label_mapping"]
        self.stage2_label_map = self.stage2_artifacts["label_mapping"]
        
        # Reverse mappings for prediction decoding
        self.stage1_idx_to_label = {v: k for k, v in self.stage1_label_map.items()}
        self.stage2_idx_to_label = {v: k for k, v in self.stage2_label_map.items()}
        
        # Label to index mapping for probability lookup
        self.stage1_label_to_idx = {k: v for k, v in self.stage1_label_map.items()}
        self.stage2_label_to_idx = {k: v for k, v in self.stage2_label_map.items()}
        
        self._loaded = True
        
        return {
            "status": "loaded",
            "stage1_warnings": self.stage1_artifacts.get("warnings", []),
            "stage2_warnings": self.stage2_artifacts.get("warnings", []),
            "stage1_errors": self.stage1_artifacts.get("errors", []),
            "stage2_errors": self.stage2_artifacts.get("errors", []),
        }
    
    def _predict_stage1(self, X: pd.DataFrame) -> np.ndarray:
        """Run Stage 1 prediction (returns class indices)."""
        scaler = self.stage1_artifacts["scaler"]
        pca = self.stage1_artifacts["pca"]
        model = self.stage1_artifacts["model"]
        
        # Ensure DataFrame with correct column names for scaler
        if hasattr(scaler, "feature_names_in_"):
            X = X[scaler.feature_names_in_]
        
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        preds = model.predict(X_pca)
        
        # Get probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pca)
        
        return preds, probs
    
    def _predict_stage2(self, X: pd.DataFrame) -> np.ndarray:
        """Run Stage 2 prediction (returns class indices)."""
        scaler = self.stage2_artifacts["scaler"]
        model = self.stage2_artifacts["model"]
        
        # Ensure DataFrame with correct column names for scaler
        if hasattr(scaler, "feature_names_in_"):
            X = X[scaler.feature_names_in_]
        
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        
        # Get probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
        
        return preds, probs
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        ground_truth: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        Run two-stage inference on a batch of flows.
        Returns list of result dicts.
        """
        if not self._loaded:
            self.load()
        
        # Preprocess for both stages
        X1, X2, prep_info = preprocess_for_inference(
            df, self.stage1_features, self.stage2_features
        )
        
        # Stage 1 prediction
        stage1_preds, stage1_probs = self._predict_stage1(X1)
        
        results = []
        stage2_indices = []
        stage2_rows = []
        # Preserve the original DataFrame index so results can be
        # cross-referenced back to the raw flow records.
        flow_indices = list(df.index)

        
        # Process Stage 1 results
        for i, pred_label in enumerate(stage1_preds):
            stage1_label = pred_label  # Already a string label
            
            result = {
                "flow_index": flow_indices[i],
                "stage1_label": stage1_label,
                "stage1_score": None,
                "stage2_label": None,
                "final_label": stage1_label,
                "ground_truth": None,
                "correct": None,
                "tier": "BENIGN" if stage1_label == "BENIGN" else "KNOWN_ATTACK",
            }
            
            # Add Stage 1 probability if available
            if stage1_probs is not None:
                # Convert label to index for probability lookup
                pred_idx = self.stage1_label_to_idx.get(stage1_label, 0)
                result["stage1_score"] = float(stage1_probs[i, pred_idx])
            
            # If attack, queue for Stage 2
            if stage1_label == "ATTACK":
                stage2_indices.append(i)
                stage2_rows.append(i)
            
            results.append(result)
        
        # Stage 2 prediction for attack flows
        if stage2_indices:
            X2_attack = X2.iloc[stage2_rows]
            stage2_preds, stage2_probs = self._predict_stage2(X2_attack)
            
            for idx, (orig_i, pred_label) in enumerate(zip(stage2_indices, stage2_preds)):
                stage2_label = pred_label  # Already a string label
                results[orig_i]["stage2_label"] = stage2_label
                results[orig_i]["final_label"] = stage2_label
                results[orig_i]["tier"] = "KNOWN_ATTACK"
                
                if stage2_probs is not None:
                    pred_idx = self.stage2_label_to_idx.get(stage2_label, 0)
                    results[orig_i]["stage2_score"] = float(stage2_probs[idx, pred_idx])
        
        # Add ground truth if provided
        if ground_truth is not None:
            gt_normalized = normalize_labels(ground_truth)
            for i, result in enumerate(results):
                if i < len(gt_normalized):
                    gt_label = gt_normalized.iloc[i]
                    result["ground_truth"] = gt_label
                    # Determine correctness
                    if result["final_label"] == "BENIGN" and gt_label.upper() == "BENIGN":
                        result["correct"] = True
                    elif result["final_label"] != "BENIGN" and gt_label.upper() != "BENIGN":
                        # For attacks, check if category matches
                        result["correct"] = (result["final_label"] == gt_label)
                    else:
                        result["correct"] = False
        
        return results
    
    def predict_one(self, row: pd.Series) -> Dict[str, Any]:
        """Run inference on a single flow."""
        df = pd.DataFrame([row])
        results = self.predict_batch(df)
        return results[0] if results else {}
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Return feature and model information."""
        return {
            "stage1_features": self.stage1_features,
            "stage2_features": self.stage2_features,
            "stage1_labels": list(self.stage1_label_map.keys()),
            "stage2_labels": list(self.stage2_label_map.keys()),
            "stage1_n_features": len(self.stage1_features) if self.stage1_features else 0,
            "stage2_n_features": len(self.stage2_features) if self.stage2_features else 0,
        }


# Global detector instance
_detector_instance = None


def get_detector() -> TwoStageDetector:
    """Get or create the global detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TwoStageDetector()
    return _detector_instance