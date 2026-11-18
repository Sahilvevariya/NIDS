"""Generate synthetic CIC-IDS2017-like data for development/testing."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


# CIC-IDS2017 feature columns (simplified subset)
CIC_FEATURES = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
]


ATTACK_TYPES = [
    "BENIGN",
    "DoS Hulk", "DoS GoldenEye", "DoS Slowloris", "DoS Slowhttptest",
    "DDoS",
    "PortScan",
    "FTP-Patator", "SSH-Patator",
    "Bot",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
    "Infiltration",
    "Heartbleed",
]


# Attack-specific parameter ranges for realistic synthetic data
ATTACK_PARAMS = {
    "BENIGN": {
        "Flow Duration": (1000, 1000000),
        "Total Fwd Packets": (1, 50),
        "Total Backward Packets": (1, 50),
        "Flow Bytes/s": (100, 1000000),
        "Flow Packets/s": (10, 1000),
        "Packet Length Mean": (60, 1500),
    },
    "DoS Hulk": {
        "Flow Duration": (1000, 50000),
        "Total Fwd Packets": (100, 1000),
        "Total Backward Packets": (1, 10),
        "Flow Bytes/s": (10000, 5000000),
        "Flow Packets/s": (500, 5000),
        "Packet Length Mean": (500, 1500),
    },
    "DoS GoldenEye": {
        "Flow Duration": (1000, 50000),
        "Total Fwd Packets": (50, 500),
        "Total Backward Packets": (1, 10),
        "Flow Bytes/s": (5000, 2000000),
        "Flow Packets/s": (200, 3000),
        "Packet Length Mean": (400, 1200),
    },
    "DoS Slowloris": {
        "Flow Duration": (100000, 5000000),
        "Total Fwd Packets": (1, 20),
        "Total Backward Packets": (1, 5),
        "Flow Bytes/s": (100, 50000),
        "Flow Packets/s": (1, 50),
        "Packet Length Mean": (200, 800),
    },
    "DoS Slowhttptest": {
        "Flow Duration": (50000, 2000000),
        "Total Fwd Packets": (1, 30),
        "Total Backward Packets": (1, 5),
        "Flow Bytes/s": (100, 100000),
        "Flow Packets/s": (1, 100),
        "Packet Length Mean": (300, 1000),
    },
    "DDoS": {
        "Flow Duration": (1000, 100000),
        "Total Fwd Packets": (100, 2000),
        "Total Backward Packets": (10, 200),
        "Flow Bytes/s": (50000, 20000000),
        "Flow Packets/s": (1000, 10000),
        "Packet Length Mean": (500, 1500),
    },
    "PortScan": {
        "Flow Duration": (100, 5000),
        "Total Fwd Packets": (1, 3),
        "Total Backward Packets": (0, 1),
        "Flow Bytes/s": (1000, 100000),
        "Flow Packets/s": (100, 10000),
        "Packet Length Mean": (40, 100),
    },
    "FTP-Patator": {
        "Flow Duration": (1000, 100000),
        "Total Fwd Packets": (5, 50),
        "Total Backward Packets": (5, 50),
        "Flow Bytes/s": (1000, 500000),
        "Flow Packets/s": (50, 500),
        "Packet Length Mean": (100, 500),
    },
    "SSH-Patator": {
        "Flow Duration": (1000, 100000),
        "Total Fwd Packets": (5, 50),
        "Total Backward Packets": (5, 50),
        "Flow Bytes/s": (1000, 500000),
        "Flow Packets/s": (50, 500),
        "Packet Length Mean": (100, 500),
    },
    "Bot": {
        "Flow Duration": (10000, 5000000),
        "Total Fwd Packets": (10, 200),
        "Total Backward Packets": (10, 200),
        "Flow Bytes/s": (500, 500000),
        "Flow Packets/s": (10, 200),
        "Packet Length Mean": (200, 800),
    },
    "Web Attack - Brute Force": {
        "Flow Duration": (1000, 500000),
        "Total Fwd Packets": (10, 200),
        "Total Backward Packets": (5, 100),
        "Flow Bytes/s": (2000, 1000000),
        "Flow Packets/s": (50, 1000),
        "Packet Length Mean": (300, 1000),
    },
    "Web Attack - XSS": {
        "Flow Duration": (1000, 200000),
        "Total Fwd Packets": (1, 10),
        "Total Backward Packets": (1, 10),
        "Flow Bytes/s": (500, 500000),
        "Flow Packets/s": (10, 200),
        "Packet Length Mean": (200, 800),
    },
    "Web Attack - Sql Injection": {
        "Flow Duration": (1000, 200000),
        "Total Fwd Packets": (1, 10),
        "Total Backward Packets": (1, 10),
        "Flow Bytes/s": (500, 500000),
        "Flow Packets/s": (10, 200),
        "Packet Length Mean": (200, 800),
    },
    "Infiltration": {
        "Flow Duration": (10000, 5000000),
        "Total Fwd Packets": (5, 100),
        "Total Backward Packets": (5, 100),
        "Flow Bytes/s": (1000, 2000000),
        "Flow Packets/s": (20, 500),
        "Packet Length Mean": (200, 1000),
    },
    "Heartbleed": {
        "Flow Duration": (1000, 50000),
        "Total Fwd Packets": (1, 5),
        "Total Backward Packets": (1, 5),
        "Flow Bytes/s": (1000, 100000),
        "Flow Packets/s": (10, 100),
        "Packet Length Mean": (100, 300),
    },
}


def generate_flow(attack_type: str, flow_id: int) -> Dict:
    """Generate a single synthetic flow record."""
    params = ATTACK_PARAMS.get(attack_type, ATTACK_PARAMS["BENIGN"])
    
    # Base values
    flow_duration = np.random.uniform(*params["Flow Duration"])
    fwd_pkts = np.random.randint(*params["Total Fwd Packets"])
    bwd_pkts = np.random.randint(*params["Total Backward Packets"])
    flow_bytes_s = np.random.uniform(*params["Flow Bytes/s"])
    flow_pkts_s = np.random.uniform(*params["Flow Packets/s"])
    pkt_len_mean = np.random.uniform(*params["Packet Length Mean"])
    
    # Derived values
    total_pkts = fwd_pkts + bwd_pkts
    fwd_len = int(pkt_len_mean * fwd_pkts * np.random.uniform(0.8, 1.2))
    bwd_len = int(pkt_len_mean * bwd_pkts * np.random.uniform(0.8, 1.2))
    
    # IPs
    src_ip = f"192.168.{np.random.randint(0,256)}.{np.random.randint(1,255)}"
    dst_ip = f"10.0.{np.random.randint(0,256)}.{np.random.randint(1,255)}"
    src_port = np.random.randint(1024, 65535)
    dst_port = np.random.choice([80, 443, 22, 21, 25, 53, 3306, 8080] + list(range(1024, 10000)))
    
    # Protocol: 6=TCP, 17=UDP
    protocol = np.random.choice([6, 17], p=[0.9, 0.1])
    
    # Timestamp
    timestamp = f"2017-07-0{np.random.randint(1,8)} {np.random.randint(0,23):02d}:{np.random.randint(0,59):02d}:{np.random.randint(0,59):02d}.{np.random.randint(0,999999):06d}"
    
    flow = {
        "Flow ID": f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}",
        "Source IP": src_ip,
        "Source Port": src_port,
        "Destination IP": dst_ip,
        "Destination Port": dst_port,
        "Protocol": protocol,
        "Timestamp": timestamp,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_len,
        "Total Length of Bwd Packets": bwd_len,
        "Fwd Packet Length Max": pkt_len_mean * np.random.uniform(1.1, 2.0),
        "Fwd Packet Length Min": pkt_len_mean * np.random.uniform(0.1, 0.5),
        "Fwd Packet Length Mean": pkt_len_mean,
        "Fwd Packet Length Std": pkt_len_mean * np.random.uniform(0.1, 0.3),
        "Bwd Packet Length Max": pkt_len_mean * np.random.uniform(1.1, 2.0),
        "Bwd Packet Length Min": pkt_len_mean * np.random.uniform(0.1, 0.5),
        "Bwd Packet Length Mean": pkt_len_mean,
        "Bwd Packet Length Std": pkt_len_mean * np.random.uniform(0.1, 0.3),
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": flow_duration / max(total_pkts, 1) * np.random.uniform(0.5, 2.0),
        "Flow IAT Std": flow_duration / max(total_pkts, 1) * np.random.uniform(0.1, 0.5),
        "Flow IAT Max": flow_duration * np.random.uniform(0.1, 0.5),
        "Flow IAT Min": flow_duration * np.random.uniform(0.001, 0.01),
        "Fwd IAT Total": flow_duration * np.random.uniform(0.5, 1.0),
        "Fwd IAT Mean": flow_duration / max(fwd_pkts, 1) * np.random.uniform(0.5, 2.0),
        "Fwd IAT Std": flow_duration / max(fwd_pkts, 1) * np.random.uniform(0.1, 0.5),
        "Fwd IAT Max": flow_duration * np.random.uniform(0.1, 0.5),
        "Fwd IAT Min": flow_duration * np.random.uniform(0.001, 0.01),
        "Bwd IAT Total": flow_duration * np.random.uniform(0.5, 1.0),
        "Bwd IAT Mean": flow_duration / max(bwd_pkts, 1) * np.random.uniform(0.5, 2.0) if bwd_pkts > 0 else 0,
        "Bwd IAT Std": flow_duration / max(bwd_pkts, 1) * np.random.uniform(0.1, 0.5) if bwd_pkts > 0 else 0,
        "Bwd IAT Max": flow_duration * np.random.uniform(0.1, 0.5) if bwd_pkts > 0 else 0,
        "Bwd IAT Min": flow_duration * np.random.uniform(0.001, 0.01) if bwd_pkts > 0 else 0,
        "Fwd PSH Flags": np.random.randint(0, 10),
        "Bwd PSH Flags": np.random.randint(0, 10),
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": fwd_pkts * 20 * np.random.uniform(0.8, 1.2),
        "Bwd Header Length": bwd_pkts * 20 * np.random.uniform(0.8, 1.2),
        "Fwd Packets/s": fwd_pkts / (flow_duration / 1e6) * np.random.uniform(0.8, 1.2) if flow_duration > 0 else 0,
        "Bwd Packets/s": bwd_pkts / (flow_duration / 1e6) * np.random.uniform(0.8, 1.2) if flow_duration > 0 else 0,
        "Min Packet Length": min(40, pkt_len_mean * 0.5),
        "Max Packet Length": max(1500, pkt_len_mean * 1.5),
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Std": pkt_len_mean * np.random.uniform(0.1, 0.3),
        "Packet Length Variance": (pkt_len_mean * np.random.uniform(0.1, 0.3)) ** 2,
        "FIN Flag Count": np.random.randint(0, 5),
        "SYN Flag Count": np.random.randint(0, 5),
        "RST Flag Count": np.random.randint(0, 3),
        "PSH Flag Count": np.random.randint(0, 10),
        "ACK Flag Count": np.random.randint(0, 20),
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": np.random.uniform(0.5, 2.0),
        "Average Packet Size": pkt_len_mean,
        "Avg Fwd Segment Size": pkt_len_mean,
        "Avg Bwd Segment Size": pkt_len_mean,
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0,
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_len,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_len,
        "Init_Win_bytes_forward": np.random.choice([65535, 64240, 5840, 5720, 2920, 1460]),
        "Init_Win_bytes_backward": np.random.choice([65535, 64240, 5840, 5720, 2920, 1460]),
        "act_data_pkt_fwd": fwd_pkts,
        "min_seg_size_forward": np.random.randint(0, 1460),
        "Active Mean": flow_duration * np.random.uniform(0.1, 0.5) if np.random.random() > 0.5 else 0,
        "Active Std": flow_duration * np.random.uniform(0.01, 0.1) if np.random.random() > 0.5 else 0,
        "Active Max": flow_duration * np.random.uniform(0.1, 0.5) if np.random.random() > 0.5 else 0,
        "Active Min": flow_duration * np.random.uniform(0.01, 0.1) if np.random.random() > 0.5 else 0,
        "Idle Mean": flow_duration * np.random.uniform(0.1, 0.5) if np.random.random() > 0.5 else 0,
        "Idle Std": flow_duration * np.random.uniform(0.01, 0.1) if np.random.random() > 0.5 else 0,
        "Idle Max": flow_duration * np.random.uniform(0.1, 0.5) if np.random.random() > 0.5 else 0,
        "Idle Min": flow_duration * np.random.uniform(0.01, 0.1) if np.random.random() > 0.5 else 0,
        "Label": attack_type,
    }
    
    return flow


def generate_dataset(
    output_dir: Path,
    n_samples: int = 5000,
    n_files: int = 4,
    random_state: int = 42,
) -> Dict:
    """Generate multiple CSV files mimicking CIC-IDS2017 structure."""
    np.random.seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define attack distributions per file (simulating different capture days)
    file_attack_configs = [
        # File 1: Mostly benign + DoS
        {"BENIGN": 0.70, "DoS Hulk": 0.10, "DoS GoldenEye": 0.05, "DoS Slowloris": 0.05, "DoS Slowhttptest": 0.05, "DDoS": 0.05},
        # File 2: Benign + PortScan + Brute Force
        {"BENIGN": 0.60, "PortScan": 0.20, "FTP-Patator": 0.05, "SSH-Patator": 0.05, "Web Attack - Brute Force": 0.10},
        # File 3: Benign + Web attacks + Bot
        {"BENIGN": 0.65, "Web Attack - XSS": 0.10, "Web Attack - Sql Injection": 0.05, "Bot": 0.10, "Infiltration": 0.10},
        # File 4: Mixed
        {"BENIGN": 0.50, "DoS Hulk": 0.10, "DDoS": 0.10, "PortScan": 0.10, "Bot": 0.10, "Heartbleed": 0.10},
    ]
    
    file_info = {}
    samples_per_file = n_samples // n_files
    
    for i in range(min(n_files, len(file_attack_configs))):
        config = file_attack_configs[i]
        filename = f"cicids2017_synthetic_day{i+1}.csv"
        filepath = output_dir / filename
        
        rows = []
        for _ in range(samples_per_file):
            # Sample attack type based on config
            attack_type = np.random.choice(list(config.keys()), p=list(config.values()))
            flow = generate_flow(attack_type, len(rows))
            rows.append(flow)
        
        df = pd.DataFrame(rows)
        # Shuffle
        df = df.sample(frac=1, random_state=random_state + i).reset_index(drop=True)
        df.to_csv(filepath, index=False)
        
        file_info[filename] = {
            "rows": len(df),
            "label_distribution": df["Label"].value_counts().to_dict(),
        }
        print(f"Generated {filename}: {len(df)} rows")
    
    return file_info


def generate_demo_sample(output_path: Path, n_samples: int = 2000, random_state: int = 42) -> Dict:
    """Generate a demo sample CSV."""
    np.random.seed(random_state)
    
    # Balanced distribution for demo
    attack_types = [a for a in ATTACK_TYPES if a != "BENIGN"]
    n_benign = n_samples // 2
    n_per_attack = max(1, (n_samples - n_benign) // len(attack_types))
    
    rows = []
    
    # Benign
    for _ in range(n_benign):
        rows.append(generate_flow("BENIGN", len(rows)))
    
    # Attacks
    for attack in attack_types:
        for _ in range(n_per_attack):
            rows.append(generate_flow(attack, len(rows)))
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    info = {
        "total_rows": len(df),
        "label_distribution": df["Label"].value_counts().to_dict(),
    }
    
    print(f"Generated demo sample: {len(df)} rows")
    print(f"Label distribution: {info['label_distribution']}")
    
    return info


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    demo_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/demo_sample.csv")
    
    print("Generating synthetic CIC-IDS2017 dataset...")
    generate_dataset(data_dir, n_samples=10000, n_files=4)
    print("\nGenerating demo sample...")
    generate_demo_sample(demo_path, n_samples=2000)
    print("Done!")