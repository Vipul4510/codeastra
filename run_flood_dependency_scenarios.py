"""
UrbanPulse Flood Dependency Scenario Evaluation Script (Phase 6)
=================================================================
Executes multi-scenario simulations across Mumbai Western Suburbs corridor assets.
Generates structured dependency evaluations and graph-edge outputs.

CRITICAL DISCLAIMER:
These scenarios are synthetic simulations designed strictly for software engineering
validation and rule integrity testing. They do NOT represent validated operational
hydrological forecasts or real-time flood predictions.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dependency_rules import (
    DependencyType,
    DependencyStatus,
    SeverityLevel,
    FloodRuleConfig
)
from flood_dependency_engine import FloodDependencyEngine


def run_all_scenarios():
    print("=" * 78)
    print("URBANPULSE: FLOOD INTELLIGENCE & INFRASTRUCTURE DEPENDENCY ANALYSIS (PHASE 6)")
    print("=" * 78)
    print("DISCLAIMER: Synthetic software simulations for algorithmic validation.")
    print("Do not present results as validated real-world flood prediction performance.\n")

    # 1. Load Real Asset & Dependency Catalog from CSVs
    assets_csv = os.path.join(CURRENT_DIR, "infrastructure_assets.csv")
    deps_csv = os.path.join(CURRENT_DIR, "asset_dependencies.csv")

    assets = []
    if os.path.exists(assets_csv):
        df_assets = pd.read_csv(assets_csv)
        assets = df_assets.to_dict(orient="records")
        print(f"[Init] Loaded {len(assets)} infrastructure assets from {os.path.basename(assets_csv)}.")

    dependencies = []
    if os.path.exists(deps_csv):
        df_deps = pd.read_csv(deps_csv)
        dependencies = df_deps.to_dict(orient="records")
        print(f"[Init] Loaded {len(dependencies)} asset dependencies from {os.path.basename(deps_csv)}.\n")

    engine = FloodDependencyEngine(assets=assets, dependencies=dependencies)

    results_summary: List[Dict[str, Any]] = []
    graph_edges: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Scenario 1: Andheri Subway Monsoon Inundation (Cascading Pluvial Chain)
    # -------------------------------------------------------------------------
    print("-" * 78)
    print("[SCENARIO 1: SIMULATION] Andheri Subway Cascading Pluvial Inundation")
    print("-" * 78)
    # Ensure local connectivity in simulation
    engine.load_dependencies([
        {"from_asset_id": "AST-00027", "to_asset_id": "AST-00026", "relationship_type": "UPSTREAM_OF", "relationship_status": "VERIFIED", "confidence_score": 0.95},
        {"from_asset_id": "AST-00026", "to_asset_id": "AST-00028", "relationship_type": "SERVES_ROAD", "relationship_status": "VERIFIED", "confidence_score": 0.92},
        {"from_asset_id": "AST-00028", "to_asset_id": "ROUTE-ANDHERI-SUBWAY", "relationship_type": "IMPACTS_ROUTE", "relationship_status": "VERIFIED", "confidence_score": 0.96}
    ])

    rain_reading_s1 = {
        "sensor_id": "AST-00027",  # Andheri rain gauge
        "timestamp": "2026-07-15T14:00:00+05:30",
        "rainfall_15min_mm": 16.5,
        "rainfall_1hr_mm": 58.0,
        "data_quality": "VALID",
        "sensor_health": "HEALTHY"
    }
    water_reading_s1 = {
        "sensor_id": "AST-00026",  # Andheri Subway
        "timestamp": "2026-07-15T14:25:00+05:30",
        "water_level_cm": 58.5,
        "water_level_change_cm": 11.2,
        "data_quality": "VALID",
        "sensor_health": "HEALTHY"
    }
    road_obs_s1 = {
        "asset_id": "AST-00028",
        "observed_time": "2026-07-15T14:30:00+05:30",
        "water_depth_cm": 52.0,
        "road_passability": "CLOSED"
    }
    traffic_s1 = {
        "route_id": "ROUTE-ANDHERI-SUBWAY",
        "timestamp": "2026-07-15T14:35:00+05:30",
        "speed_drop_pct": 85.0,
        "congestion_delay_min": 45.0,
        "traffic_status": "IMPASSABLE"
    }

    # Step 1a: Rainfall -> Water Rise
    res_1a = engine.analyze_rainfall_to_water_rise(rain_reading_s1, water_reading_s1)
    print(f"  [1a] Rainfall -> Water Rise: {res_1a.status.value} (Conf: {res_1a.confidence_score}, Sev: {res_1a.severity_level.value})")
    print(f"       Explanation: {res_1a.explanation}")

    # Step 1b: Water Rise -> Road Flooding
    res_1b = engine.analyze_water_level_to_road_waterlogging(water_reading_s1, road_obs_s1)
    print(f"  [1b] Water Rise -> Road Flooding: {res_1b.status.value} (Conf: {res_1b.confidence_score}, Sev: {res_1b.severity_level.value})")

    # Step 1c: Road Flooding -> Traffic Closure
    res_1c = engine.analyze_road_waterlogging_to_traffic_slowdown(road_obs_s1, traffic_s1)
    print(f"  [1c] Road Flooding -> Traffic Closure: {res_1c.status.value} (Conf: {res_1c.confidence_score}, Sev: {res_1c.severity_level.value})")

    for r in (res_1a, res_1b, res_1c):
        results_summary.append({"scenario": "Scenario 1: Andheri Subway", "result": r.__dict__})
        edge = engine.convert_analysis_to_graph_edge(r)
        if edge:
            graph_edges.append(edge)

    # -------------------------------------------------------------------------
    # Scenario 2: Invariant 4 Test (High Water Level Alone without Blockage)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[SCENARIO 2: SIMULATION] Milan Subway High Water Level (Invariant 4 Check)")
    print("-" * 78)
    # Verify drainage connection exists so connectivity check passes,
    # thereby specifically testing Invariant 4 (blockage cannot be inferred from water level alone)
    engine.load_dependencies([
        {"from_asset_id": "AST-00024", "to_asset_id": "AST-00026", "relationship_type": "DRAINS_TO", "relationship_status": "VERIFIED", "confidence_score": 0.94}
    ])
    drain_reading_s2 = {
        "sensor_id": "AST-00024",  # Andheri drain
        "timestamp": "2026-07-15T15:00:00+05:30",
        "blockage_probability": 0.05,  # Clear / low probability
        "drainage_capacity_pct": 85.0
    }
    sump_reading_s2 = {
        "sensor_id": "AST-00026",
        "timestamp": "2026-07-15T15:15:00+05:30",
        "water_level_cm": 65.0  # High water level!
    }
    res_2 = engine.analyze_drain_blockage_to_accumulation(drain_reading_s2, sump_reading_s2)
    print(f"  [2] Drain Blockage Inference: {res_2.status.value} (Conf: {res_2.confidence_score})")
    print(f"      Explanation: {res_2.explanation}")
    print(f"      Alternative Hypotheses: {res_2.alternative_hypotheses}")
    results_summary.append({"scenario": "Scenario 2: Invariant 4 Safeguard", "result": res_2.__dict__})

    # -------------------------------------------------------------------------
    # Scenario 3: Invariant 5 Test (Power Outage vs Pump Failure Verification)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[SCENARIO 3: SIMULATION] Versova Pump Station Power Failure (Invariant 5 Check)")
    print("-" * 78)
    # 3a. Case A: Outage on UNRELATED feeder (Must be REFUTED)
    unrelated_feeder = {
        "asset_id": "AST-00025",  # Borivali feeder
        "timestamp": "2026-07-15T16:00:00+05:30",
        "status": "OFFLINE"
    }
    pump_station_versova = {
        "asset_id": "AST-00006",  # Versova pump station
        "timestamp": "2026-07-15T16:10:00+05:30",
        "pump_status": "FAILED"
    }
    res_3a = engine.analyze_power_to_pump_failure(unrelated_feeder, pump_station_versova)
    print(f"  [3a] Unrelated Feeder Outage -> Pump Failure: {res_3a.status.value} (Conf: {res_3a.confidence_score})")
    print(f"       Safeguard Enforced: {res_3a.explanation}")

    # 3b. Case B: Outage on VERIFIED feeder (AST-00004 powers AST-00006)
    engine.load_dependencies([
        {"from_asset_id": "AST-00004", "to_asset_id": "AST-00006", "relationship_type": "POWERED_BY", "relationship_status": "VERIFIED", "confidence_score": 0.98}
    ])
    verified_feeder = {
        "asset_id": "AST-00004",  # Versova power feeder
        "timestamp": "2026-07-15T16:00:00+05:30",
        "status": "OFFLINE"
    }
    res_3b = engine.analyze_power_to_pump_failure(verified_feeder, pump_station_versova)
    print(f"  [3b] Verified Feeder Outage -> Pump Failure: {res_3b.status.value} (Conf: {res_3b.confidence_score}, Sev: {res_3b.severity_level.value})")
    print(f"       Explanation: {res_3b.explanation}")

    results_summary.append({"scenario": "Scenario 3a: Unrelated Power Feeder", "result": res_3a.__dict__})
    results_summary.append({"scenario": "Scenario 3b: Verified Power Circuit", "result": res_3b.__dict__})
    edge_3b = engine.convert_analysis_to_graph_edge(res_3b)
    if edge_3b:
        graph_edges.append(edge_3b)

    # -------------------------------------------------------------------------
    # Scenario 4: Contradictory Traffic Clearance Testing
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[SCENARIO 4: SIMULATION] Contradictory Traffic Telemetry Isolation")
    print("-" * 78)
    road_obs_s4 = {
        "asset_id": "AST-00028",
        "observed_time": "2026-07-15T17:00:00+05:30",
        "water_depth_cm": 40.0,
        "road_passability": "CLOSED"
    }
    traffic_s4_freeflow = {
        "route_id": "ROUTE-ANDHERI-SUBWAY",
        "timestamp": "2026-07-15T17:10:00+05:30",
        "speed_drop_pct": 0.0,
        "congestion_delay_min": 0.0,
        "traffic_status": "FREE_FLOW"
    }
    res_4 = engine.analyze_road_waterlogging_to_traffic_slowdown(road_obs_s4, traffic_s4_freeflow)
    print(f"  [4] Contradiction Analysis: {res_4.status.value} (Conf: {res_4.confidence_score})")
    print(f"      Explanation: {res_4.explanation}")
    results_summary.append({"scenario": "Scenario 4: Contradiction Isolation", "result": res_4.__dict__})

    # Export Results
    edges_out_path = os.path.join(CURRENT_DIR, "flood_dependency_graph_edges.json")
    with open(edges_out_path, "w", encoding="utf-8") as f:
        json.dump(graph_edges, f, indent=2)
    print(f"\n[Export] Saved {len(graph_edges)} validated dependency graph edges to: {edges_out_path}")

    eval_out_path = os.path.join(CURRENT_DIR, "flood_dependency_scenario_results.json")
    # Clean non-serializable objects
    serializable_summary = []
    for item in results_summary:
        res_copy = dict(item["result"])
        res_copy["dependency_type"] = res_copy["dependency_type"].value
        res_copy["status"] = res_copy["status"].value
        res_copy["severity_level"] = res_copy["severity_level"].value
        res_copy["supporting_evidence"] = [ev.__dict__ for ev in res_copy.get("supporting_evidence", [])]
        res_copy["contradictory_evidence"] = [ev.__dict__ for ev in res_copy.get("contradictory_evidence", [])]
        serializable_summary.append({"scenario": item["scenario"], "result": res_copy})

    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"[Export] Saved structured scenario evaluation results to: {eval_out_path}")

    print("\n" + "=" * 78)
    print("PHASE 6 SIMULATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_scenarios()
