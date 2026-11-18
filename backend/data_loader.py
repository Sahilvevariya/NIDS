"""Data loading utilities with caching."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from config import RAW_DATA_DIR, DEMO_SAMPLE_PATH, LEAKAGE_COLUMNS


@st.cache_data(ttl=3600, show_spinner="Loading demo sample...")
def load_demo_sample() -> pd.DataFrame:
    """Load the pre-exported demo sample CSV."""
    if not DEMO_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Demo sample not found at {DEMO_SAMPLE_PATH}. "
            f"Run the export script first."
        )
    return pd.read_csv(DEMO_SAMPLE_PATH, low_memory=False)


def discover_csv_files(data_dir: Path) -> List[Path]:
    """Discover all CSV files in a directory."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


@st.cache_data(ttl=3600, show_spinner="Loading raw CSV files...")
def load_raw_csvs(data_dir: Path, max_files: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and concatenate multiple CSV files.
    Returns (combined_dataframe, file_info_dict).
    """
    csv_files = discover_csv_files(data_dir)
    if max_files:
        csv_files = csv_files[:max_files]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    file_info = {}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Normalize columns
            df.columns = df.columns.str.strip()
            # Normalize labels
            if "Label" in df.columns:
                df["Label"] = df["Label"].astype(str).str.strip()
            
            dfs.append(df)
            file_info[f.name] = {
                "rows": len(df),
                "cols": len(df.columns),
                "labels": df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
            }
        except Exception as e:
            file_info[f.name] = {"error": str(e)}
    
    if not dfs:
        raise ValueError("No valid CSV files could be loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined, file_info


def create_demo_sample(
    raw_data_dir: Path,
    output_path: Path,
    n_samples: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Create a demo sample from raw CSV files.
    Stratifies by label to ensure representation.
    """
    combined, file_info = load_raw_csvs(raw_data_dir)
    
    if "Label" not in combined.columns:
        raise ValueError("No 'Label' column found in data")
    
    # Normalize labels
    combined["Label"] = combined["Label"].astype(str).str.strip()
    
    # Stratified sample
    sampled_parts = []
    for label, group in combined.groupby("Label"):
        n = min(len(group), max(1, n_samples // combined["Label"].nunique()))
        sampled = group.sample(n=n, random_state=random_state)
        sampled_parts.append(sampled)
    
    demo = pd.concat(sampled_parts, ignore_index=True)
    demo = demo.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(output_path, index=False)
    
    info = {
        "total_rows": len(demo),
        "label_distribution": demo["Label"].value_counts().to_dict(),
        "source_files": list(file_info.keys()),
    }
    
    return info


def get_sample_modes(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Get row indices for different sample modes."""
    modes = {
        "all": list(range(len(df))),
        "random": [],
        "benign": [],
        "attack": [],
    }
    
    if "Label" in df.columns:
        labels = df["Label"].astype(str).str.strip().str.upper()
        modes["benign"] = df[labels == "BENIGN"].index.tolist()
        modes["attack"] = df[labels != "BENIGN"].index.tolist()
    
    return modes


def filter_by_attack_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter dataframe to specific attack category."""
    if "Label" not in df.columns:
        return df
    return df[df["Label"].astype(str).str.strip() == category]


def get_attack_categories(df: pd.DataFrame) -> List[str]:
    """Get list of unique attack categories (excluding BENIGN)."""
    if "Label" not in df.columns:
        return []
    labels = df["Label"].astype(str).str.strip()
    return sorted([l for l in labels.unique() if l.upper() != "BENIGN"])