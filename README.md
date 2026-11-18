# Network Intrusion Detection & Attack Classification System

**Machine Learning-Based Network Security Analysis Using CIC-IDS2017**

---

## 📋 What This Is

A **dataset-based replay prototype** of a two-stage Machine Learning Network Intrusion Detection System trained on the CIC-IDS2017 dataset. This is an academic demonstration project showing how ML models can classify network traffic as benign or malicious, and identify specific attack types.

### What This Is NOT
- ❌ Not a live Network Intrusion Detection System (NIDS)
- ❌ Not packet capture or real-time monitoring
- ❌ Not a production security tool
- ❌ Not a replacement for firewalls, WAFs, or endpoint protection

---

## 🏗️ Architecture

### Two-Stage ML Pipeline

```
Network Flow Record
        │
        ▼
┌───────────────────┐
│  Stage 1: Binary  │  ──► BENIGN → Final Result: BENIGN
│  Classification   │
│  (RF + PCA)       │
└───────────────────┘
        │
        ▼ ATTACK
┌───────────────────┐
│  Stage 2: Multi-  │  ──► Attack Category (14 classes)
│  class Classification  │
│  (RF + Scaler)    │
└───────────────────┘
```

**Stage 1** — Binary Intrusion Detection
- Algorithm: Random Forest (100 estimators, max_depth=20, balanced class weights)
- Preprocessing: StandardScaler → PCA (25 components)
- Input: 79 numeric flow features
- Output: BENIGN or ATTACK

**Stage 2** — Attack Type Classification
- Algorithm: Random Forest (100 estimators, max_depth=20, balanced class weights)
- Preprocessing: StandardScaler
- Input: 79 numeric flow features (attack records only)
- Output: 1 of 14 attack categories

### Attack Categories (14 classes)
- DoS: Hulk, GoldenEye, Slowloris, Slowhttptest
- DDoS
- PortScan
- Brute Force: FTP-Patator, SSH-Patator
- Bot
- Web Attacks: Brute Force, XSS, SQL Injection
- Infiltration
- Heartbleed

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (tested on 3.13)
- Windows/Linux/macOS

### Installation

```bash
# Clone or navigate to project
cd Network-Intrusion-Detection

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Generate Synthetic Data (for demo)
```bash
# Generates synthetic CIC-IDS2017-like data + demo sample
python generate_synthetic_data.py data/raw data/demo_sample.csv
```

### Train Models
```bash
# Train Stage 1 (Binary Classification)
python train_stage1.py data/raw

# Train Stage 2 (Attack Classification)
python train_stage2.py data/raw
```

### Run Dashboard
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
Network-Intrusion-Detection/
├── app.py                      # Streamlit dashboard entry point
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── generate_synthetic_data.py  # Synthetic data generator
├── train_stage1.py             # Stage 1 training script
├── train_stage2.py             # Stage 2 training script
├── models/
│   ├── stage1/
│   │   ├── binary_model.joblib     # Trained RF model
│   │   ├── scaler.joblib           # StandardScaler
│   │   ├── pca.joblib              # PCA transformer
│   │   ├── feature_columns.json    # Feature names & order
│   │   └── label_mapping.json      # Label encoding
│   ├── stage2/
│   │   ├── attack_model.joblib     # Trained RF model
│   │   ├── scaler.joblib           # StandardScaler
│   │   ├── feature_columns.json    # Feature names & order
│   │   └── label_mapping.json      # Label encoding
│   └── metadata.json               # Training metadata & hashes
├── data/
│   ├── raw/                        # Raw CSV files (CIC-IDS2017)
│   └── demo_sample.csv             # Pre-sampled demo data
├── backend/
│   ├── __init__.py
│   ├── model_loader.py     # Artifact loading & validation
│   ├── preprocessing.py    # Feature preprocessing pipeline
│   ├── detector.py         # Two-stage inference engine
│   ├── data_loader.py      # Data loading & sampling
│   ├── evaluation.py       # Metrics computation
│   ├── metrics.py          # Dashboard KPI calculations
│   └── replay_engine.py    # Dataset replay simulation
└── results/                    # Output directory
```

---

## 🖥️ Dashboard Pages

### 1. Overview Dashboard
- KPI cards: total flows, benign/attack counts, attack rate
- Prediction timeline chart
- Attack type distribution
- Recent predictions table

### 2. Traffic Analysis
- Select individual flow by index
- View key flow features (duration, packets, bytes, rates)
- Stage 1 & Stage 2 predictions
- Ground truth comparison (Evaluation mode)

### 3. Detection Simulator
- **Sample Modes**: Random, Benign-only, Attack-only, Specific attack category
- **Batch Size**: 10–500 flows per step
- **Interval**: 1–10 seconds between batches
- **Controls**: Start, Pause, Reset, Single Step
- Live-updating predictions table

### 4. Attack Alerts
- Only flows predicted as attacks
- Filter by attack category
- Verdict tags: Correct Detection, False Positive, Missed Attack, Wrong Category
- Export to CSV

### 5. Model Performance
- Overall accuracy, precision, recall, F1
- Stage 1 (Binary) metrics with confusion matrix
- Stage 2 (Multiclass) metrics with per-class table
- Rare-class warnings (low support = high variance)

---

## ⚙️ Configuration

Key settings in `config.py`:
- `STAGE1_PCA_COMPONENTS`: PCA dimensions (default: 25)
- `STAGE2_MAX_SAMPLES_PER_CLASS`: Training cap per attack class (default: 10000)
- `DEFAULT_BATCH_SIZE`, `DEFAULT_INTERVAL_SECONDS`: Replay defaults
- `COLORS`: UI color scheme

---

## 📊 Results & Limitations

### Training Performance (Synthetic Data)
| Stage | Train Acc | Test Acc | Notes |
|-------|-----------|----------|-------|
| Stage 1 (Binary) | 99.84% | 93.45% | PCA variance: 86.89% |
| Stage 2 (Multiclass) | 99.74% | 87.37% | Macro F1: 77% |

### Known Limitations
1. **Synthetic Data**: Demo uses generated data mimicking CIC-IDS2017 structure, not real captures
2. **Random Split**: Evaluation uses random train/test split — may not reflect temporal generalization
3. **Rare Classes**: Some attack types have very few test samples (<10), making F1 estimates unreliable
4. **No Payload Inspection**: Model uses flow statistics only — cannot detect payload-based attacks (SQLi, XSS) reliably
5. **Version Sensitivity**: scikit-learn version mismatch warnings may appear; pin versions for exact reproducibility
6. **No Live Capture**: This is a dataset replay prototype only

### Stage 1 Accuracy Note
The original project reported 99.78% test accuracy with confusion matrix [[23962, 38], [66, 11590]]. 
Our reproduction on synthetic data gives 93.45%. The discrepancy is due to:
- Different data distribution (synthetic vs real CIC-IDS2017)
- Different random splits
- PCA variance difference (86.89% vs reported 95.77%)

---

## 🔧 Extending the System

The codebase includes **extension hooks** for future enhancements:

### Layer 3: Anomaly Detection
```python
# In detector.py - add after Stage 1 BENIGN prediction
if stage1_label == "BENIGN":
    anomaly_score = isolation_forest.score_samples(X)
    if anomaly_score > threshold:
        tier = "UNKNOWN_ANOMALOUS"
```

### Layer 1: Signature Detection (Suricata/Zeek)
```python
# backend/sources/signatures.py
def check_signatures(flow):
    # Return alerts in same result schema
    return {"source": "SIGNATURE", "final_label": "...", ...}
```

### Layer 2: Device Baselines
```python
# backend/baselines.py
def compute_zscore(flow, device_baseline):
    z = (flow.feature - baseline.mean) / baseline.std
    return z
```

### Central Risk Engine
```python
# backend/risk.py
def compute_risk(ml_result, anomaly_score, signature_alerts, zscore):
    # Combine into unified risk score
    return risk_score
```

---

## 📝 Reproducing Metrics

```bash
# Full evaluation on demo sample
python -c "
from backend.data_loader import load_demo_sample
from backend.detector import get_detector
from backend.evaluation import compute_session_metrics

df = load_demo_sample()
detector = get_detector()
detector.load()
preds = detector.predict_batch(df, df['Label'])
metrics = compute_session_metrics(preds)
import json
print(json.dumps(metrics, indent=2))
"
```

---

## 📚 References

- **CIC-IDS2017 Dataset**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Original Paper**: Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
- **scikit-learn**: https://scikit-learn.org/
- **Streamlit**: https://streamlit.io/

---

## ⚖️ License

Academic project for educational purposes. Not for production use.

---

## 👥 Authors

Network Intrusion Detection Project — ML-based NIDS using CIC-IDS2017

---

## 📌 Important Disclaimer

> **This is a dataset-based replay prototype for academic demonstration only.**
> 
> It does not perform live packet capture, real-time monitoring, or production-grade intrusion detection. 
> The models are trained on synthetic data mimicking CIC-IDS2017 and should not be deployed in real security environments.
> For actual network security, use established tools like Suricata, Zeek, Snort, or commercial NIDS solutions.