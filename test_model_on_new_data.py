"""
UrbanPulse: Model Inference Test on New / Upcoming Data
========================================================
Tests whether the trained Phase 4 ML pipeline generalises to unseen,
freshly-generated observation pairs.

What this script does:
  1. Loads saved ML artifacts (calibrated model, scaler, label encoder, features).
  2. Generates 500 realistic *new* pairwise observation samples that the model
     has NEVER seen during training — covering all 5 relationship classes plus
     edge-case stress scenarios.
  3. Runs inference through the full pipeline (feature engineering -> scaling -> prediction).
  4. Evaluates accuracy, per-class precision/recall/F1, confusion matrix,
     confidence distribution analysis, and edge-case pass/fail report.
  5. Prints a clean summary and saves results to `new_data_test_results.json`.

Usage:
    python test_model_on_new_data.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Ensure project imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from feature_engineering import PairwiseFeatureExtractor


# ==============================================================================
# Section 1: Load Saved ML Artifacts
# ==============================================================================

def load_ml_artifacts(artifacts_dir: str) -> Dict[str, Any]:
    """Loads all serialized ML artifacts from disk."""
    print("  Loading calibrated_classifier.joblib ...")
    model = joblib.load(os.path.join(artifacts_dir, "calibrated_classifier.joblib"))

    print("  Loading feature_scaler.joblib ...")
    scaler = joblib.load(os.path.join(artifacts_dir, "feature_scaler.joblib"))

    print("  Loading label_encoder.joblib ...")
    le = joblib.load(os.path.join(artifacts_dir, "label_encoder.joblib"))

    print("  Loading feature_names.json ...")
    with open(os.path.join(artifacts_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    print("  Loading target_classes.json ...")
    with open(os.path.join(artifacts_dir, "target_classes.json"), "r") as f:
        target_classes = json.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
        "target_classes": target_classes,
    }


# ==============================================================================
# Section 2: Generate Realistic New (Unseen) Observation Pairs
# ==============================================================================

# Mumbai Western Suburbs localities for realistic place names
LOCALITIES = [
    "Andheri West", "Bandra West", "Goregaon East", "Malad West",
    "Kandivali East", "Borivali West", "Versova", "Jogeshwari East",
    "Dahisar East", "Vile Parle West", "Santacruz East", "Khar West"
]

EVENT_TYPES = [
    "ROAD_WATERLOGGING", "WATER_LEVEL_RISE", "SUBWAY_INUNDATION",
    "TRAFFIC_SLOWDOWN", "ROAD_CLOSURE", "DRAIN_BLOCKAGE",
    "HEAVY_RAINFALL", "MANHOLE_OVERFLOW", "PUMP_FAILURE",
    "TRANSIT_DELAY", "ROAD_ACCIDENT", "GENERAL_INCIDENT"
]

# Causal pairs where one event type can reasonably cause the other
CAUSAL_EVENT_PAIRS = [
    ("HEAVY_RAINFALL", "ROAD_WATERLOGGING"),
    ("DRAIN_BLOCKAGE", "WATER_LEVEL_RISE"),
    ("ROAD_WATERLOGGING", "TRAFFIC_SLOWDOWN"),
    ("SUBWAY_INUNDATION", "ROAD_CLOSURE"),
    ("PUMP_FAILURE", "WATER_LEVEL_RISE"),
    ("ROAD_ACCIDENT", "TRAFFIC_SLOWDOWN"),
]


def _random_mumbai_coord():
    """Generate a random lat/lon within Mumbai Western Suburbs bounding box."""
    lat = np.random.uniform(19.05, 19.26)
    lon = np.random.uniform(72.80, 72.87)
    return lat, lon


def _nearby_coord(lat, lon, radius_km):
    """Offset a coordinate by approximately `radius_km`."""
    delta_deg = radius_km / 111.0  # rough degree conversion
    new_lat = lat + np.random.uniform(-delta_deg, delta_deg)
    new_lon = lon + np.random.uniform(-delta_deg, delta_deg)
    return new_lat, new_lon


def generate_new_observation_pairs(n_per_class: int = 100, seed: int = 2026) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Generates brand-new synthetic pairwise observation data that mimics
    realistic Mumbai Western Suburbs incident reporting patterns.

    Returns:
        df_pairs: DataFrame with pairwise feature columns matching training format.
        df_obs:   Fake observations DataFrame for feature enrichment.
        y_true:   Ground truth labels for evaluation.
    """
    np.random.seed(seed)

    pairs_data = []
    obs_records = {}
    true_labels = []
    pair_counter = 0

    def _create_obs(obs_id, lat, lon, event_type, locality, text, reliability=None):
        if reliability is None:
            reliability = np.random.uniform(0.40, 0.98)
        obs_records[obs_id] = {
            "observation_id": obs_id,
            "source_type": np.random.choice(["CITIZEN_REPORT", "SENSOR", "SOCIAL_POST", "FIELD_TEAM"]),
            "event_type": event_type,
            "event_time": (datetime(2026, 10, 13) + timedelta(minutes=np.random.randint(-1440, 0))).isoformat(),
            "latitude": lat,
            "longitude": lon,
            "location_text": locality,
            "location_accuracy_m": np.random.uniform(5, 100),
            "source_reliability_score": reliability,
            "verification_status": np.random.choice(["VERIFIED", "UNVERIFIED", "PENDING"]),
            "is_duplicate_candidate": False,
            "text_original": text,
            "text_normalized": text.lower().strip(),
            "source_id": f"SRC-NEW-{np.random.randint(1000, 9999)}",
        }
        return obs_records[obs_id]

    # ---- Class 1: DUPLICATE (near-identical re-reports) ----
    for i in range(n_per_class):
        pair_counter += 1
        lat, lon = _random_mumbai_coord()
        loc = np.random.choice(LOCALITIES)
        etype = np.random.choice(EVENT_TYPES)
        base_text = f"Heavy waterlogging reported near {loc} junction, traffic at standstill"

        obs_a_id = f"NEW-OBS-DUP-{pair_counter:04d}-A"
        obs_b_id = f"NEW-OBS-DUP-{pair_counter:04d}-B"

        _create_obs(obs_a_id, lat, lon, etype, loc, base_text)
        lat_b, lon_b = _nearby_coord(lat, lon, np.random.uniform(0.01, 0.12))
        text_b = base_text  # near-identical text
        _create_obs(obs_b_id, lat_b, lon_b, etype, loc, text_b)

        pairs_data.append({
            "pair_id": f"NEW-PAIR-{pair_counter:05d}",
            "observation_id_a": obs_a_id,
            "observation_id_b": obs_b_id,
            "time_diff_minutes": np.random.uniform(0.5, 15.0),
            "geo_distance_km": np.random.uniform(0.01, 0.12),
            "text_similarity": np.random.uniform(0.85, 0.99),
            "same_locality": 1,
            "same_source": np.random.choice([0, 1]),
            "reliability_product": np.random.uniform(0.30, 0.90),
        })
        true_labels.append("DUPLICATE")

    # ---- Class 2: SAME_INCIDENT (same event, different reporters) ----
    for i in range(n_per_class):
        pair_counter += 1
        lat, lon = _random_mumbai_coord()
        loc = np.random.choice(LOCALITIES)
        etype = np.random.choice(EVENT_TYPES)

        obs_a_id = f"NEW-OBS-SAME-{pair_counter:04d}-A"
        obs_b_id = f"NEW-OBS-SAME-{pair_counter:04d}-B"

        _create_obs(obs_a_id, lat, lon, etype, loc,
                    f"Severe flooding at {loc}, water level rising fast")
        lat_b, lon_b = _nearby_coord(lat, lon, np.random.uniform(0.05, 0.90))
        _create_obs(obs_b_id, lat_b, lon_b, etype, loc,
                    f"Water accumulation near {loc}, vehicles stranded")

        pairs_data.append({
            "pair_id": f"NEW-PAIR-{pair_counter:05d}",
            "observation_id_a": obs_a_id,
            "observation_id_b": obs_b_id,
            "time_diff_minutes": np.random.uniform(2.0, 40.0),
            "geo_distance_km": np.random.uniform(0.05, 0.90),
            "text_similarity": np.random.uniform(0.65, 0.92),
            "same_locality": 1,
            "same_source": 0,
            "reliability_product": np.random.uniform(0.30, 0.85),
        })
        true_labels.append("SAME_INCIDENT")

    # ---- Class 3: RELATED_BUT_DISTINCT (causally linked, different events) ----
    for i in range(n_per_class):
        pair_counter += 1
        lat, lon = _random_mumbai_coord()
        loc = np.random.choice(LOCALITIES)
        cause_type, effect_type = CAUSAL_EVENT_PAIRS[i % len(CAUSAL_EVENT_PAIRS)]

        obs_a_id = f"NEW-OBS-REL-{pair_counter:04d}-A"
        obs_b_id = f"NEW-OBS-REL-{pair_counter:04d}-B"

        _create_obs(obs_a_id, lat, lon, cause_type, loc,
                    f"{cause_type.replace('_', ' ').title()} observed at {loc}")
        lat_b, lon_b = _nearby_coord(lat, lon, np.random.uniform(0.50, 2.50))
        loc_b = np.random.choice(LOCALITIES)
        _create_obs(obs_b_id, lat_b, lon_b, effect_type, loc_b,
                    f"{effect_type.replace('_', ' ').title()} reported near {loc_b}")

        pairs_data.append({
            "pair_id": f"NEW-PAIR-{pair_counter:05d}",
            "observation_id_a": obs_a_id,
            "observation_id_b": obs_b_id,
            "time_diff_minutes": np.random.uniform(30.0, 120.0),
            "geo_distance_km": np.random.uniform(0.50, 2.50),
            "text_similarity": np.random.uniform(0.35, 0.65),
            "same_locality": int(loc == loc_b),
            "same_source": 0,
            "reliability_product": np.random.uniform(0.25, 0.75),
        })
        true_labels.append("RELATED_BUT_DISTINCT")

    # ---- Class 4: UNRELATED (completely different events far apart) ----
    for i in range(n_per_class):
        pair_counter += 1
        lat_a, lon_a = _random_mumbai_coord()
        lat_b, lon_b = _random_mumbai_coord()
        loc_a = np.random.choice(LOCALITIES)
        loc_b = np.random.choice([l for l in LOCALITIES if l != loc_a])
        etype_a = np.random.choice(EVENT_TYPES)
        etype_b = np.random.choice([e for e in EVENT_TYPES if e != etype_a])

        obs_a_id = f"NEW-OBS-UNR-{pair_counter:04d}-A"
        obs_b_id = f"NEW-OBS-UNR-{pair_counter:04d}-B"

        _create_obs(obs_a_id, lat_a, lon_a, etype_a, loc_a,
                    f"{etype_a.replace('_', ' ').title()} at {loc_a}")
        _create_obs(obs_b_id, lat_b, lon_b, etype_b, loc_b,
                    f"{etype_b.replace('_', ' ').title()} at {loc_b}")

        geo_dist = np.random.uniform(3.0, 8.0)
        pairs_data.append({
            "pair_id": f"NEW-PAIR-{pair_counter:05d}",
            "observation_id_a": obs_a_id,
            "observation_id_b": obs_b_id,
            "time_diff_minutes": np.random.uniform(200.0, 800.0),
            "geo_distance_km": geo_dist,
            "text_similarity": np.random.uniform(0.05, 0.30),
            "same_locality": 0,
            "same_source": 0,
            "reliability_product": np.random.uniform(0.20, 0.70),
        })
        true_labels.append("UNRELATED")

    # ---- Class 5: CONTRADICTORY (conflicting reports about same location) ----
    for i in range(n_per_class):
        pair_counter += 1
        lat, lon = _random_mumbai_coord()
        loc = np.random.choice(LOCALITIES)

        obs_a_id = f"NEW-OBS-CON-{pair_counter:04d}-A"
        obs_b_id = f"NEW-OBS-CON-{pair_counter:04d}-B"

        _create_obs(obs_a_id, lat, lon, "ROAD_WATERLOGGING", loc,
                    f"Road completely flooded and closed near {loc}, impassable")
        lat_b, lon_b = _nearby_coord(lat, lon, np.random.uniform(0.05, 0.60))
        _create_obs(obs_b_id, lat_b, lon_b, "ROAD_WATERLOGGING", loc,
                    f"Road is clear and open near {loc}, traffic moving normally")

        pairs_data.append({
            "pair_id": f"NEW-PAIR-{pair_counter:05d}",
            "observation_id_a": obs_a_id,
            "observation_id_b": obs_b_id,
            "time_diff_minutes": np.random.uniform(5.0, 30.0),
            "geo_distance_km": np.random.uniform(0.05, 0.60),
            "text_similarity": np.random.uniform(0.30, 0.60),
            "same_locality": 1,
            "same_source": 0,
            "reliability_product": np.random.uniform(0.25, 0.70),
        })
        true_labels.append("CONTRADICTORY")

    df_pairs = pd.DataFrame(pairs_data)
    df_obs = pd.DataFrame(list(obs_records.values()))
    y_true = np.array(true_labels)

    return df_pairs, df_obs, y_true


# ==============================================================================
# Section 3: Generate Edge-Case Stress Test Scenarios
# ==============================================================================

def generate_edge_case_scenarios() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Creates hand-crafted edge-case scenarios to stress-test the model's
    robustness on tricky, borderline inputs.
    """
    np.random.seed(9999)
    scenarios = []
    obs_records = {}
    pairs_data = []

    def _add_obs(obs_id, lat, lon, etype, loc, text, reliability=0.7):
        obs_records[obs_id] = {
            "observation_id": obs_id,
            "source_type": "FIELD_TEAM",
            "event_type": etype,
            "event_time": "2026-10-13T10:00:00",
            "latitude": lat,
            "longitude": lon,
            "location_text": loc,
            "location_accuracy_m": 20.0,
            "source_reliability_score": reliability,
            "verification_status": "VERIFIED",
            "is_duplicate_candidate": False,
            "text_original": text,
            "text_normalized": text.lower(),
            "source_id": f"SRC-EDGE-{len(obs_records)+1}",
        }

    # Edge Case 1: Extremely close in space/time but contradictory text
    _add_obs("EDGE-01-A", 19.12, 72.84, "ROAD_WATERLOGGING", "Andheri West",
             "Road is completely submerged, 2 feet of standing water")
    _add_obs("EDGE-01-B", 19.1202, 72.8401, "ROAD_WATERLOGGING", "Andheri West",
             "Road is completely dry, no water at all, traffic flowing freely")
    pairs_data.append({
        "pair_id": "EDGE-PAIR-001",
        "observation_id_a": "EDGE-01-A", "observation_id_b": "EDGE-01-B",
        "time_diff_minutes": 3.0, "geo_distance_km": 0.02,
        "text_similarity": 0.35, "same_locality": 1, "same_source": 0,
        "reliability_product": 0.49,
    })
    scenarios.append({
        "id": "EDGE-001",
        "name": "Very close spatially but contradictory claims",
        "expected_class": "CONTRADICTORY",
        "difficulty": "HIGH",
        "description": "Two reports at same location, same time, but one says 'flooded' and other says 'dry'."
    })

    # Edge Case 2: Same text copy-paste but VERY far apart (should be UNRELATED, not DUPLICATE)
    _add_obs("EDGE-02-A", 19.06, 72.82, "TRAFFIC_SLOWDOWN", "Bandra West",
             "Major traffic jam due to waterlogging, avoid this area")
    _add_obs("EDGE-02-B", 19.24, 72.86, "TRAFFIC_SLOWDOWN", "Dahisar East",
             "Major traffic jam due to waterlogging, avoid this area")
    pairs_data.append({
        "pair_id": "EDGE-PAIR-002",
        "observation_id_a": "EDGE-02-A", "observation_id_b": "EDGE-02-B",
        "time_diff_minutes": 5.0, "geo_distance_km": 20.5,
        "text_similarity": 1.00, "same_locality": 0, "same_source": 0,
        "reliability_product": 0.50,
    })
    scenarios.append({
        "id": "EDGE-002",
        "name": "Identical text but geographically impossible",
        "expected_class": "UNRELATED",
        "difficulty": "HIGH",
        "description": "Copy-pasted viral message from 2 cities - model must reject as unrelated despite text_sim=1.0."
    })

    # Edge Case 3: Borderline SAME_INCIDENT vs RELATED_BUT_DISTINCT
    _add_obs("EDGE-03-A", 19.15, 72.84, "HEAVY_RAINFALL", "Goregaon East",
             "Intense rainfall in Goregaon, roads getting flooded")
    _add_obs("EDGE-03-B", 19.155, 72.845, "ROAD_WATERLOGGING", "Goregaon East",
             "Waterlogging on Goregaon SV road, ankle-deep water")
    pairs_data.append({
        "pair_id": "EDGE-PAIR-003",
        "observation_id_a": "EDGE-03-A", "observation_id_b": "EDGE-03-B",
        "time_diff_minutes": 25.0, "geo_distance_km": 0.65,
        "text_similarity": 0.52, "same_locality": 1, "same_source": 0,
        "reliability_product": 0.60,
    })
    scenarios.append({
        "id": "EDGE-003",
        "name": "Rainfall -> Waterlogging causal chain (borderline)",
        "expected_class": "RELATED_BUT_DISTINCT",
        "difficulty": "MEDIUM",
        "description": "Rain report and subsequent waterlogging - physically related but distinct incident types."
    })

    # Edge Case 4: Extremely low reliability sources
    _add_obs("EDGE-04-A", 19.18, 72.83, "ROAD_CLOSURE", "Malad West",
             "Road closed at Malad flyover", reliability=0.10)
    _add_obs("EDGE-04-B", 19.181, 72.831, "ROAD_CLOSURE", "Malad West",
             "Malad flyover road blocked completely", reliability=0.12)
    pairs_data.append({
        "pair_id": "EDGE-PAIR-004",
        "observation_id_a": "EDGE-04-A", "observation_id_b": "EDGE-04-B",
        "time_diff_minutes": 8.0, "geo_distance_km": 0.14,
        "text_similarity": 0.88, "same_locality": 1, "same_source": 0,
        "reliability_product": 0.012,
    })
    scenarios.append({
        "id": "EDGE-004",
        "name": "Low-reliability near-duplicate",
        "expected_class": "DUPLICATE",
        "difficulty": "MEDIUM",
        "description": "Very similar reports from unreliable sources - model should still detect near-duplicate."
    })

    # Edge Case 5: Large time gap, same location (stale report)
    _add_obs("EDGE-05-A", 19.20, 72.85, "SUBWAY_INUNDATION", "Kandivali East",
             "Subway flooded at Kandivali station")
    _add_obs("EDGE-05-B", 19.20, 72.85, "SUBWAY_INUNDATION", "Kandivali East",
             "Kandivali subway still flooded")
    pairs_data.append({
        "pair_id": "EDGE-PAIR-005",
        "observation_id_a": "EDGE-05-A", "observation_id_b": "EDGE-05-B",
        "time_diff_minutes": 600.0, "geo_distance_km": 0.00,
        "text_similarity": 0.80, "same_locality": 1, "same_source": 0,
        "reliability_product": 0.55,
    })
    scenarios.append({
        "id": "EDGE-005",
        "name": "Same location but 10-hour time gap",
        "expected_class": "UNRELATED",
        "difficulty": "MEDIUM",
        "description": "Reports 10 hours apart at same subway - likely different flood events, not same incident."
    })

    df_pairs = pd.DataFrame(pairs_data)
    df_obs = pd.DataFrame(list(obs_records.values()))

    return df_pairs, df_obs, scenarios


# ==============================================================================
# Section 4: Run Full Inference Pipeline
# ==============================================================================

def run_inference(
    artifacts: Dict[str, Any],
    df_pairs: pd.DataFrame,
    df_obs: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the complete inference pipeline:
      features -> model -> predicted classes + probabilities
    """
    model = artifacts["model"]
    le = artifacts["label_encoder"]

    # Extract features using the same pipeline as training
    extractor = PairwiseFeatureExtractor(df_obs)
    X_new = extractor.extract_features(df_pairs, df_obs)

    # Predict class labels and probabilities
    y_pred_encoded = model.predict(X_new)
    y_pred_proba = model.predict_proba(X_new)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    return y_pred_labels, y_pred_proba, y_pred_encoded


# ==============================================================================
# Section 5: Main Execution
# ==============================================================================

def main():
    start_time = time.time()

    print("=" * 78)
    print("  URBANPULSE: MODEL INFERENCE TEST ON NEW / UPCOMING DATA")
    print("=" * 78)

    # ── Step 1: Load artifacts ──
    artifacts_dir = os.path.join(CURRENT_DIR, "ml_artifacts")
    print(f"\n[Step 1] Loading saved ML artifacts from: {artifacts_dir}")

    if not os.path.exists(artifacts_dir):
        print("  [X] ERROR: ml_artifacts directory not found! Train the model first.")
        sys.exit(1)

    artifacts = load_ml_artifacts(artifacts_dir)
    le = artifacts["label_encoder"]
    class_names = artifacts["target_classes"]
    print(f"  [OK] All artifacts loaded. Target classes: {class_names}")

    # ── Step 2: Generate new unseen data ──
    N_PER_CLASS = 100
    print(f"\n[Step 2] Generating {N_PER_CLASS * 5} NEW unseen observation pairs...")
    print(f"         ({N_PER_CLASS} per class x 5 classes)")

    df_pairs, df_obs, y_true = generate_new_observation_pairs(n_per_class=N_PER_CLASS)
    print(f"  [OK] Generated {len(df_pairs)} pairs from {len(df_obs)} unique observations.")
    print(f"  [OK] True label distribution:")
    for lbl, cnt in pd.Series(y_true).value_counts().sort_index().items():
        print(f"      {lbl:<25}: {cnt}")

    # ── Step 3: Run inference ──
    print(f"\n[Step 3] Running inference through the calibrated ML pipeline...")
    y_pred, y_proba, y_pred_encoded = run_inference(artifacts, df_pairs, df_obs)
    y_true_encoded = le.transform(y_true)

    # ── Step 4: Evaluate results ──
    print(f"\n[Step 4] Evaluation Results on New Data")
    print("-" * 60)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  +---------------------------------------------+")
    print(f"  |  OVERALL ACCURACY :  {acc:.4f}  ({acc*100:.1f}%)          |")
    print(f"  |  MACRO F1-SCORE   :  {f1_macro:.4f}                   |")
    print(f"  |  WEIGHTED F1-SCORE:  {f1_weighted:.4f}                   |")
    print(f"  |  MACRO PRECISION  :  {p_macro:.4f}                   |")
    print(f"  |  MACRO RECALL     :  {r_macro:.4f}                   |")
    print(f"  +---------------------------------------------+")

    # Per-class report
    print(f"\n  Per-Class Classification Report:")
    print("  " + "-" * 72)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(f"\n  Confusion Matrix:")
    print("  " + "-" * 72)
    title_col = 'True \\ Predicted'
    header = f"  {title_col:<23}" + "".join(f"{c[:12]:>14}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"  {row_name:<23}" + "".join(f"{cm[i][j]:>14}" for j in range(len(class_names)))
        print(row_str)

    # ── Step 5: Confidence Distribution Analysis ──
    print(f"\n[Step 5] Prediction Confidence Analysis")
    print("-" * 60)

    max_probs = np.max(y_proba, axis=1)
    correct_mask = (y_pred == y_true)

    print(f"  Overall confidence stats:")
    print(f"    Mean confidence  : {np.mean(max_probs):.4f}")
    print(f"    Median confidence: {np.median(max_probs):.4f}")
    print(f"    Min confidence   : {np.min(max_probs):.4f}")
    print(f"    Max confidence   : {np.max(max_probs):.4f}")

    # Confidence by correctness
    if correct_mask.any():
        print(f"  Correct predictions   - mean confidence: {np.mean(max_probs[correct_mask]):.4f}")
    if (~correct_mask).any():
        print(f"  Incorrect predictions - mean confidence: {np.mean(max_probs[~correct_mask]):.4f}")

    # Confidence thresholds analysis
    print(f"\n  Accuracy at different confidence thresholds:")
    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        mask = max_probs >= threshold
        if mask.any():
            acc_at_thresh = accuracy_score(y_true[mask], y_pred[mask])
            coverage = mask.sum() / len(y_true) * 100
            print(f"    Conf >= {threshold:.2f}: Accuracy = {acc_at_thresh:.4f} | Coverage = {coverage:.1f}% ({mask.sum()}/{len(y_true)})")

    # ── Step 6: Edge-Case Stress Tests ──
    print(f"\n[Step 6] Edge-Case Stress Tests")
    print("-" * 60)

    df_edge_pairs, df_edge_obs, edge_scenarios = generate_edge_case_scenarios()
    y_edge_pred, y_edge_proba, _ = run_inference(artifacts, df_edge_pairs, df_edge_obs)

    edge_results = []
    passed = 0
    for i, scenario in enumerate(edge_scenarios):
        predicted = y_edge_pred[i]
        expected = scenario["expected_class"]
        confidence = np.max(y_edge_proba[i])
        is_pass = predicted == expected

        status = "[PASS]" if is_pass else "[FAIL]"
        if is_pass:
            passed += 1

        print(f"\n  [{scenario['id']}] {scenario['name']}")
        print(f"     Difficulty : {scenario['difficulty']}")
        print(f"     Expected   : {expected}")
        print(f"     Predicted  : {predicted} (confidence: {confidence:.4f})")
        print(f"     Result     : {status}")
        if not is_pass:
            print(f"     >> Note   : {scenario['description']}")

        edge_results.append({
            "scenario_id": scenario["id"],
            "name": scenario["name"],
            "expected": expected,
            "predicted": predicted,
            "confidence": round(float(confidence), 4),
            "passed": is_pass,
        })

    print(f"\n  Edge-Case Summary: {passed}/{len(edge_scenarios)} passed")

    # ── Step 7: Save results ──
    results_path = os.path.join(CURRENT_DIR, "new_data_test_results.json")
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_pairs": len(df_pairs),
            "per_class": N_PER_CLASS,
            "generation_seed": 2026,
            "is_completely_new_data": True,
        },
        "overall_metrics": {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
        },
        "per_class": {},
        "confidence_analysis": {
            "mean": round(float(np.mean(max_probs)), 4),
            "median": round(float(np.median(max_probs)), 4),
            "min": round(float(np.min(max_probs)), 4),
            "max": round(float(np.max(max_probs)), 4),
            "correct_mean": round(float(np.mean(max_probs[correct_mask])), 4) if correct_mask.any() else None,
            "incorrect_mean": round(float(np.mean(max_probs[~correct_mask])), 4) if (~correct_mask).any() else None,
        },
        "confusion_matrix": cm.tolist(),
        "edge_case_results": edge_results,
        "edge_case_pass_rate": f"{passed}/{len(edge_scenarios)}",
        "model_verdict": "PASS - Model generalises to new data" if acc >= 0.70 else "NEEDS ATTENTION - Accuracy below 70% on new data"
    }

    # Per-class breakdown
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )
    for i, cls in enumerate(class_names):
        results["per_class"][cls] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1_score": round(float(f1_cls[i]), 4),
            "support": int(sup_cls[i]),
        }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 78}")
    print(f"  VERDICT: {results['model_verdict']}")
    print(f"  Results saved to: {results_path}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
