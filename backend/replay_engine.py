"""Replay engine for dataset-based traffic simulation."""
import time
import random
from typing import Dict, List, Optional, Iterator, Generator
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_BATCH_SIZE, DEFAULT_INTERVAL_SECONDS,
    MIN_BATCH_SIZE, MAX_BATCH_SIZE, MIN_INTERVAL, MAX_INTERVAL
)
from backend.data_loader import get_sample_modes, filter_by_attack_category, get_attack_categories


class ReplayEngine:
    """Generates batches of flows for simulated replay."""
    
    def __init__(self, dataframe: pd.DataFrame, random_state: int = 42):
        self.df = dataframe.copy()
        self.random_state = random_state
        self.rng = random.Random(random_state)
        
        # State
        self.mode = "random"
        self.batch_size = DEFAULT_BATCH_SIZE
        self.interval = DEFAULT_INTERVAL_SECONDS
        self.attack_category = None
        self.indices = []
        self.position = 0
        self.running = False
        self.paused = False
        self.total_processed = 0
        self.session_results = []
        
        # Initialize indices
        self._update_indices()
    
    def _update_indices(self):
        """Update the index list based on current mode."""
        modes = get_sample_modes(self.df)
        
        if self.mode == "benign":
            self.indices = modes["benign"]
        elif self.mode == "attack":
            self.indices = modes["attack"]
        elif self.attack_category:
            filtered = filter_by_attack_category(self.df, self.attack_category)
            self.indices = filtered.index.tolist()
        else:
            # Random or all
            self.indices = modes["all"]
        
        # Shuffle if random mode
        if self.mode == "random":
            self.rng.shuffle(self.indices)
        
        self.position = 0
    
    def set_mode(self, mode: str):
        """Set sampling mode: 'random', 'benign', 'attack', or 'category'."""
        self.mode = mode
        self.attack_category = None
        self._update_indices()
    
    def set_attack_category(self, category: str):
        """Set specific attack category to replay."""
        self.mode = "category"
        self.attack_category = category
        self._update_indices()
    
    def set_batch_size(self, size: int):
        """Set batch size (clamped to valid range)."""
        self.batch_size = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, size))
    
    def set_interval(self, seconds: int):
        """Set replay interval (clamped to valid range)."""
        self.interval = max(MIN_INTERVAL, min(MAX_INTERVAL, seconds))
    
    def set_random_state(self, seed: int):
        """Set random seed for reproducibility."""
        self.random_state = seed
        self.rng = random.Random(seed)
        self._update_indices()
    
    def get_next_batch(self) -> Optional[pd.DataFrame]:
        """Get the next batch of flows. Returns None if exhausted."""
        if self.position >= len(self.indices):
            return None
        
        end_pos = min(self.position + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.position:end_pos]
        self.position = end_pos
        self.total_processed += len(batch_indices)
        
        return self.df.loc[batch_indices].copy()
    
    def reset(self):
        """Reset replay state."""
        self.position = 0
        self.total_processed = 0
        self.session_results = []
        self._update_indices()
    
    def get_progress(self) -> Dict:
        """Get current progress info."""
        return {
            "position": self.position,
            "total": len(self.indices),
            "processed": self.total_processed,
            "remaining": len(self.indices) - self.position,
            "percent": (self.position / len(self.indices) * 100) if self.indices else 0,
            "mode": self.mode,
            "attack_category": self.attack_category,
            "batch_size": self.batch_size,
            "interval": self.interval,
        }
    
    def is_exhausted(self) -> bool:
        """Check if all indices have been processed."""
        return self.position >= len(self.indices)


def create_replay_engine(dataframe: pd.DataFrame) -> ReplayEngine:
    """Factory function to create a ReplayEngine (for Streamlit caching)."""
    return ReplayEngine(dataframe)


# Streamlit session state management
REPLAY_STATE_KEY = "replay_engine"
REPLAY_RESULTS_KEY = "replay_results"
REPLAY_CONFIG_KEY = "replay_config"


def init_replay_state(dataframe: pd.DataFrame) -> ReplayEngine:
    """Initialize or get replay engine from session state."""
    if REPLAY_STATE_KEY not in st.session_state:
        st.session_state[REPLAY_STATE_KEY] = create_replay_engine(dataframe)
        st.session_state[REPLAY_RESULTS_KEY] = []
        st.session_state[REPLAY_CONFIG_KEY] = {
            "mode": "random",
            "batch_size": DEFAULT_BATCH_SIZE,
            "interval": DEFAULT_INTERVAL_SECONDS,
            "attack_category": None,
            "random_state": 42,
        }
    return st.session_state[REPLAY_STATE_KEY]


def get_replay_engine() -> Optional[ReplayEngine]:
    """Get replay engine from session state."""
    return st.session_state.get(REPLAY_STATE_KEY)


def get_replay_results() -> List:
    """Get accumulated replay results."""
    return st.session_state.get(REPLAY_RESULTS_KEY, [])


def add_replay_results(results: List[Dict]):
    """Add results to session state."""
    if REPLAY_RESULTS_KEY not in st.session_state:
        st.session_state[REPLAY_RESULTS_KEY] = []
    st.session_state[REPLAY_RESULTS_KEY].extend(results)


def clear_replay_results():
    """Clear replay results."""
    st.session_state[REPLAY_RESULTS_KEY] = []


def update_replay_config(config: Dict):
    """Update replay configuration."""
    st.session_state[REPLAY_CONFIG_KEY] = config
    engine = get_replay_engine()
    if engine:
        engine.set_mode(config.get("mode", "random"))
        engine.set_batch_size(config.get("batch_size", DEFAULT_BATCH_SIZE))
        engine.set_interval(config.get("interval", DEFAULT_INTERVAL_SECONDS))
        engine.set_random_state(config.get("random_state", 42))
        if config.get("attack_category"):
            engine.set_attack_category(config["attack_category"])