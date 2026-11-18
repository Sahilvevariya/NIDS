"""KPI and metrics calculations for dashboard."""
from typing import Dict, List, Any
import pandas as pd


def compute_kpis(results: List[Dict]) -> Dict[str, Any]:
    """Compute key performance indicators from results."""
    if not results:
        return {
            "total_flows": 0,
            "benign_count": 0,
            "attack_count": 0,
            "attack_rate": 0.0,
            "unique_attack_types": 0,
            "attack_types": [],
        }
    
    total = len(results)
    benign = sum(1 for r in results if r.get("final_label") == "BENIGN")
    attack = total - benign
    
    attack_types = set()
    for r in results:
        if r.get("stage2_label"):
            attack_types.add(r["stage2_label"])
    
    return {
        "total_flows": total,
        "benign_count": benign,
        "attack_count": attack,
        "attack_rate": (attack / total * 100) if total > 0 else 0.0,
        "unique_attack_types": len(attack_types),
        "attack_types": sorted(list(attack_types)),
    }


def get_recent_predictions(results: List[Dict], n: int = 20) -> List[Dict]:
    """Get the N most recent predictions."""
    return results[-n:] if results else []


def get_attack_distribution(results: List[Dict]) -> Dict[str, int]:
    """Get distribution of predicted attack types."""
    dist = {}
    for r in results:
        label = r.get("final_label", "UNKNOWN")
        if label != "BENIGN":
            dist[label] = dist.get(label, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


def get_prediction_timeline(results: List[Dict], window: int = 50) -> List[Dict]:
    """Get cumulative attack count over time for timeline chart."""
    timeline = []
    cumulative_attacks = 0
    cumulative_total = 0
    
    for i, r in enumerate(results):
        cumulative_total += 1
        if r.get("final_label") != "BENIGN":
            cumulative_attacks += 1
        
        if i % max(1, len(results) // window) == 0 or i == len(results) - 1:
            timeline.append({
                "flow_index": i,
                "cumulative_total": cumulative_total,
                "cumulative_attacks": cumulative_attacks,
                "attack_rate": (cumulative_attacks / cumulative_total * 100) if cumulative_total > 0 else 0,
            })
    
    return timeline


def get_alerts(results: List[Dict], evaluation_mode: bool = False) -> List[Dict]:
    """Get attack alerts (only predicted attacks)."""
    alerts = []
    for r in results:
        if r.get("final_label") != "BENIGN":
            alert = {
                "flow_index": r.get("flow_index"),
                "predicted_attack": r.get("final_label"),
                "stage1_prediction": r.get("stage1_label"),
                "stage2_prediction": r.get("stage2_label"),
            }
            
            if evaluation_mode and r.get("ground_truth") is not None:
                alert["ground_truth"] = r.get("ground_truth")
                alert["correct"] = r.get("correct")
                
                # Tag correctness
                gt = str(r.get("ground_truth", "")).upper()
                pred = str(r.get("final_label", "")).upper()
                if gt == "BENIGN" and pred != "BENIGN":
                    alert["verdict"] = "FALSE_POSITIVE"
                elif gt != "BENIGN" and pred == "BENIGN":
                    alert["verdict"] = "MISSED_ATTACK"
                elif gt == pred:
                    alert["verdict"] = "CORRECT_DETECTION"
                else:
                    alert["verdict"] = "WRONG_CATEGORY"
            else:
                alert["verdict"] = "ATTACK_DETECTED"
            
            alerts.append(alert)
    
    return alerts


def filter_alerts_by_category(alerts: List[Dict], category: str) -> List[Dict]:
    """Filter alerts by attack category."""
    if category == "All":
        return alerts
    return [a for a in alerts if a.get("predicted_attack") == category]


def compute_session_summary(results: List[Dict]) -> Dict:
    """Compute a summary of the current session."""
    if not results:
        return {
            "duration_seconds": 0,
            "flows_per_second": 0,
            "peak_attack_rate": 0,
        }
    
    # Simple summary - in a real replay, we'd track actual time
    total = len(results)
    attacks = sum(1 for r in results if r.get("final_label") != "BENIGN")
    
    # Calculate running attack rate in windows of 10
    window_size = 10
    peak_rate = 0
    for i in range(0, total, window_size):
        window = results[i:i+window_size]
        window_attacks = sum(1 for r in window if r.get("final_label") != "BENIGN")
        rate = (window_attacks / len(window) * 100) if window else 0
        peak_rate = max(peak_rate, rate)
    
    return {
        "total_flows": total,
        "total_attacks": attacks,
        "overall_attack_rate": (attacks / total * 100) if total > 0 else 0,
        "peak_attack_rate": peak_rate,
        "unique_attacks_detected": len(set(
            r.get("final_label") for r in results if r.get("final_label") != "BENIGN"
        )),
    }