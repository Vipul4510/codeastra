"""
UrbanPulse Small Scenario Graph Pipeline (Phase 5)
===================================================
Constructs, validates, queries, and serializes the Andheri Subway Flood Scenario graph.
Exports:
- sample_scenario_graph.json
- graph_scenario_query_results.json
"""

import os
import sys
import json
import time
from typing import Dict, Any
import networkx as nx

# Ensure project path is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from graph_validation import NodeType, EdgeType, GraphValidator
from graph_builder import UrbanPulseEvidenceGraph
from graph_queries import UrbanPulseGraphQueries


def build_andheri_flood_scenario() -> UrbanPulseEvidenceGraph:
    """
    Constructs a validated, rich multimodal flooding scenario for Andheri Subway.
    """
    eg = UrbanPulseEvidenceGraph(name="Andheri_Subway_Flood_Scenario")

    # 1. Incident Node
    eg.add_node(
        node_id="INC-ANDHERI-FLOOD-01",
        node_type=NodeType.INCIDENT,
        label="Andheri Subway Severe Monsoon Inundation",
        locality="Andheri West",
        latitude=19.1197,
        longitude=72.8464,
        incident_category="SUBWAY_INUNDATION",
        severity="CRITICAL",
        status="ACTIVE",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 2. Multimodal Evidence Nodes
    # 2a. Citizen Eyewitness Report
    eg.add_node(
        node_id="OBS-CITIZEN-01",
        node_type=NodeType.OBSERVATION,
        label="Citizen Eyewitness Report (Andheri Station)",
        locality="Andheri West",
        latitude=19.1195,
        longitude=72.8460,
        source_modality="CITIZEN_REPORT",
        source_reliability=0.75,
        text_raw="Andheri subway waterlogged 55cm, cars stranded, traffic halted completely",
        timestamp="2026-07-15T14:25:00+05:30"
    )

    # 2b. Sensor Telemetry Reading
    eg.add_node(
        node_id="SNS-TELEMETRY-01",
        node_type=NodeType.SENSOR,
        label="Andheri Water Level Ultrasonic Telemetry",
        locality="Andheri West",
        latitude=19.1197,
        longitude=72.8464,
        sensor_type="ULTRASONIC_WATER_LEVEL",
        source_reliability=0.96,
        measured_depth_cm=58.5,
        status="ALERT_CRITICAL",
        timestamp="2026-07-15T14:28:00+05:30"
    )

    # 2c. Viral Social Post (Original)
    eg.add_node(
        node_id="POST-TWITTER-01",
        node_type=NodeType.SOCIAL_POST,
        label="Tweet: Viral Flood Alert SV Road",
        locality="Andheri West",
        latitude=19.1200,
        longitude=72.8470,
        platform="TWITTER",
        source_reliability=0.65,
        text_raw="Massive flooding at Andheri subway avoid SV road completely! #MumbaiRains",
        timestamp="2026-07-15T14:29:00+05:30"
    )

    # 2d. Bot Repost / Retweet (Single evidence chain demonstration)
    eg.add_node(
        node_id="POST-TWITTER-02",
        node_type=NodeType.SOCIAL_POST,
        label="Retweet: @trafficbot repost",
        locality="Andheri West",
        latitude=19.1200,
        longitude=72.8470,
        platform="TWITTER",
        source_reliability=0.50,
        text_raw="RT @citizen: Massive flooding at Andheri subway avoid SV road completely! #MumbaiRains",
        is_repost=True,
        repost_parent_id="POST-TWITTER-01",
        timestamp="2026-07-15T14:45:00+05:30"
    )

    # 2e. Contradictory Civic Assertion (Direct conflict demonstration)
    eg.add_node(
        node_id="OBS-CONFLICT-01",
        node_type=NodeType.OBSERVATION,
        label="Civic Notice: Subway Passable Claim",
        locality="Andheri West",
        latitude=19.1196,
        longitude=72.8463,
        source_modality="CITIZEN_REPORT",
        source_reliability=0.45,
        text_raw="Andheri subway is completely clear and open for vehicles, normal traffic flow",
        timestamp="2026-07-15T14:32:00+05:30"
    )

    # 3. Infrastructure Assets
    eg.add_node(
        node_id="AST-00026",
        node_type=NodeType.INFRASTRUCTURE_ASSET,
        label="Andheri Subway Underpass Arterial Culvert",
        locality="Andheri West",
        latitude=19.1197,
        longitude=72.8464,
        asset_type="SUBWAY_UNDERPASS",
        elevation_m=4.2,
        design_capacity_lps=1200.0,
        asset_status="SUBMERGED"
    )

    eg.add_node(
        node_id="AST-00024",
        node_type=NodeType.INFRASTRUCTURE_ASSET,
        label="Andheri Stormwater Drain Box Culvert",
        locality="Andheri West",
        latitude=19.1180,
        longitude=72.8450,
        asset_type="DRAIN",
        design_capacity_lps=3500.0,
        asset_status="BLOCKED_OVERFLOW"
    )

    eg.add_node(
        node_id="AST-00052",
        node_type=NodeType.INFRASTRUCTURE_ASSET,
        label="Andheri Western Railway Suburban Station",
        locality="Andheri West",
        latitude=19.1198,
        longitude=72.8468,
        asset_type="RAIL_STATION",
        elevation_m=7.5,
        asset_status="OPERATIONAL_DELAYED"
    )

    # 4. Routes and Transit Corridors
    eg.add_node(
        node_id="ROUTE-ANDHERI-SUBWAY",
        node_type=NodeType.ROUTE,
        label="Andheri Subway East-West Link",
        locality="Andheri",
        origin="Andheri West SV Road",
        destination="Andheri East WEH",
        travel_mode="ROAD_VEHICULAR",
        status="CLOSED"
    )

    eg.add_node(
        node_id="ROUTE-SV-ROAD",
        node_type=NodeType.ROUTE,
        label="Swami Vivekananda (SV) Road Arterial",
        locality="Andheri West",
        origin="Bandra Junction",
        destination="Dahisar Checknaka",
        travel_mode="ROAD_VEHICULAR",
        status="SEVERE_CONGESTION"
    )


    eg.add_node(
        node_id="TRANSIT-WESTERN-RAILWAY",
        node_type=NodeType.TRANSIT_SERVICE,
        label="Western Railway Suburban Slow Line Corridor",
        locality="Western Suburbs",
        origin="Churchgate",
        destination="Dahanu Road",
        travel_mode="COMMUTER_RAIL",
        status="RUNNING_SLOW_15MIN_DELAY"
    )

    # -------------------------------------------------------------------------
    # 5. Connect Validated Relationships (Evidence & Identity Edges)
    # -------------------------------------------------------------------------

    # 5a. Corroborating Evidence -> Incident
    eg.add_edge(
        source_id="OBS-CITIZEN-01",
        target_id="INC-ANDHERI-FLOOD-01",
        relationship_type=EdgeType.SUPPORTS,
        confidence=0.88,
        evidence_ids=["OBS-CITIZEN-01"],
        rule_or_model_version="RuleEngine_v3.1",
        explanation="Eyewitness citizen confirmation of 55cm water depth at subway entrance.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    eg.add_edge(
        source_id="SNS-TELEMETRY-01",
        target_id="INC-ANDHERI-FLOOD-01",
        relationship_type=EdgeType.SUPPORTS,
        confidence=0.96,
        evidence_ids=["SNS-TELEMETRY-01"],
        rule_or_model_version="RuleEngine_v3.1",
        explanation="Hardware water level sensor confirms 58.5cm standing water in underpass.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    eg.add_edge(
        source_id="POST-TWITTER-01",
        target_id="INC-ANDHERI-FLOOD-01",
        relationship_type=EdgeType.SUPPORTS,
        confidence=0.78,
        evidence_ids=["POST-TWITTER-01"],
        rule_or_model_version="CalibratedLightGBM_v4",
        explanation="Social crowd signal confirms traffic paralysis at subway approach.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 5b. Repost Chain (Single Evidence Chain Rule)
    eg.add_edge(
        source_id="POST-TWITTER-02",
        target_id="POST-TWITTER-01",
        relationship_type=EdgeType.DUPLICATE_OF,
        confidence=0.98,
        evidence_ids=["POST-TWITTER-02", "POST-TWITTER-01"],
        rule_or_model_version="Rule_DuplicateChain",
        explanation="Near-instant viral social repost (Jaccard similarity 0.96). Treated as single evidence chain.",
        timestamp="2026-07-15T14:45:00+05:30"
    )

    # 5c. Factual Contradiction Edge (Isolated conflict)
    eg.add_edge(
        source_id="OBS-CONFLICT-01",
        target_id="OBS-CITIZEN-01",
        relationship_type=EdgeType.CONTRADICTS,
        confidence=0.85,
        evidence_ids=["OBS-CONFLICT-01", "OBS-CITIZEN-01"],
        rule_or_model_version="Rule_ContradictionMatrix",
        explanation="Factual conflict: Normal passability claim directly contradicts 55cm submersion reports.",
        timestamp="2026-07-15T14:35:00+05:30"
    )

    # -------------------------------------------------------------------------
    # 6. Physical Impact & Infrastructure Dependency Edges
    # -------------------------------------------------------------------------

    # 6a. Incident -> Asset (Physical Asset Impairment)
    eg.add_edge(
        source_id="INC-ANDHERI-FLOOD-01",
        target_id="AST-00026",
        relationship_type=EdgeType.AFFECTS_ASSET,
        confidence=0.95,
        evidence_ids=["SNS-TELEMETRY-01", "OBS-CITIZEN-01"],
        rule_or_model_version="SpatialAssetMapper_v1",
        explanation="Severe localized inundation (depth 58.5cm) submerges subway underpass barrel.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 6b. Upstream Drainage Dependency -> Underpass
    eg.add_edge(
        source_id="AST-00024",
        target_id="AST-00026",
        relationship_type=EdgeType.POSSIBLE_DEPENDENCY,
        confidence=0.82,
        evidence_ids=["AST-00024", "AST-00026"],
        rule_or_model_version="HydraulicAssetDependency_v1",
        explanation="Upstream box drain blockage prevents gravity discharge from subway sump.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 6b2. Underpass Drainage -> Railway Station Access Dependency
    eg.add_edge(
        source_id="AST-00026",
        target_id="AST-00052",
        relationship_type=EdgeType.POSSIBLE_DEPENDENCY,
        confidence=0.79,
        evidence_ids=["AST-00026", "AST-00052"],
        rule_or_model_version="HydraulicAssetDependency_v1",
        explanation="Subway culvert flooding backs up into railway station pedestrian subway access.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 6c. Asset -> Sensor Observation
    eg.add_edge(
        source_id="AST-00026",
        target_id="SNS-TELEMETRY-01",
        relationship_type=EdgeType.OBSERVED_BY,
        confidence=0.99,
        evidence_ids=["AST-00026"],
        rule_or_model_version="AssetRegistry_v1",
        explanation="Underpass is continuously monitored by ultrasonic water level gauge.",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    # 6d. Incident -> Route Impairments
    eg.add_edge(
        source_id="INC-ANDHERI-FLOOD-01",
        target_id="ROUTE-ANDHERI-SUBWAY",
        relationship_type=EdgeType.IMPACTS_ROUTE,
        confidence=0.96,
        evidence_ids=["INC-ANDHERI-FLOOD-01"],
        rule_or_model_version="RouteImpactEngine_v1",
        explanation="Underpass impassable; traffic police activated physical barriers.",
        impact_type="TOTAL_CLOSURE",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    eg.add_edge(
        source_id="INC-ANDHERI-FLOOD-01",
        target_id="ROUTE-SV-ROAD",
        relationship_type=EdgeType.IMPACTS_ROUTE,
        confidence=0.88,
        evidence_ids=["INC-ANDHERI-FLOOD-01"],
        rule_or_model_version="RouteImpactEngine_v1",
        explanation="Traffic diversion from subway creates a 2.5km gridlock bottleneck on SV Road.",
        impact_type="SEVERE_CONGESTION",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    eg.add_edge(
        source_id="INC-ANDHERI-FLOOD-01",
        target_id="TRANSIT-WESTERN-RAILWAY",
        relationship_type=EdgeType.IMPACTS_ROUTE,
        confidence=0.74,
        evidence_ids=["INC-ANDHERI-FLOOD-01"],
        rule_or_model_version="TransitCorridorLinker_v1",
        explanation="Track culvert near subway approaches water safety threshold; slow speeds enforced.",
        impact_type="SPEED_RESTRICTION_DELAY",
        timestamp="2026-07-15T14:30:00+05:30"
    )

    return eg


def run_scenario_pipeline():
    print("=" * 75)
    print("URBANPULSE: EVIDENCE GRAPH CONSTRUCTION & SCENARIO QUERIES (PHASE 5)")
    print("=" * 75)

    # 1. Build Scenario Graph
    print("\n[Step 1] Constructing Andheri Subway Flood Scenario Graph...")
    eg = build_andheri_flood_scenario()
    print(f"  - Nodes created: {eg.graph.number_of_nodes()}")
    print(f"  - Edges created: {eg.graph.number_of_edges()}")

    # 2. Audit and Validate Graph Schema
    print("\n[Step 2] Executing Strict Semantic & Schema Validation...")
    audit = GraphValidator.audit_entire_graph(eg.graph)
    print(f"  - Schema Valid: {audit['is_valid']}")
    print(f"  - Node Counts by Type: {audit['node_counts_by_type']}")
    print(f"  - Edge Counts by Type: {audit['edge_counts_by_type']}")
    print(f"  - Isolated Nodes: {audit['isolated_nodes_count']}")
    print(f"  - Contradictions Flagged: {audit['active_contradictions_count']}")
    if audit["validation_issues"]:
        for issue in audit["validation_issues"]:
            print(f"      * {issue}")

    # 3. Execute Graph Queries
    print("\n[Step 3] Executing Domain Graph Queries...")
    queries = UrbanPulseGraphQueries(eg)

    # Query 1: Incident Evidence
    q1_evidence = queries.get_incident_evidence("INC-ANDHERI-FLOOD-01")
    print("\n--- Query 1: Incident Evidence Chain ---")
    print(f"  - Incident: {q1_evidence['incident_id']}")
    print(f"  - Total Evidence Nodes: {q1_evidence['total_evidence_nodes']}")
    print(f"  - Independent Evidence Chains: {q1_evidence['independent_evidence_chains']} (Repost chain consolidated!)")
    print(f"  - Mean Confidence: {q1_evidence['mean_evidence_confidence']}")
    for ev in q1_evidence["evidence_items"]:
        print(f"      * [{ev['node_type']}] {ev['evidence_id']}: conf={ev['confidence']} ({ev['relationship_to_incident']}) is_repost={ev['is_repost_or_duplicate']}")

    # Query 2: Contradictions
    q2_contra = queries.get_incident_contradictions("INC-ANDHERI-FLOOD-01")
    print("\n--- Query 2: Contradiction Isolation ---")
    print(f"  - Has Contradictions: {q2_contra['has_contradictions']}")
    print(f"  - Conflicting Claims Count: {q2_contra['contradiction_count']}")
    for c in q2_contra["conflicts"]:
        print(f"      * Conflict between {c['conflicting_node_a']} and {c['conflicting_node_b']} ({c['node_b_label']}): {c['conflict_explanation']}")

    # Query 3: Connected Infrastructure & Dependencies
    q3_infra = queries.get_connected_infrastructure("INC-ANDHERI-FLOOD-01", max_hops=2)
    print("\n--- Query 3: Affected Infrastructure & Dependencies ---")
    print(f"  - Direct Assets Affected: {q3_infra['direct_assets_count']}")
    print(f"  - Total Connected Assets (2 hops): {q3_infra['total_connected_assets']}")
    for ast in q3_infra["connected_assets"]:
        print(f"      * [Hop {ast['hop_distance']}] {ast['asset_id']} ({ast['asset_type']}) - {ast['label']} ({ast['connection_reason']})")

    # Query 4: Reachable Downstream Route Impacts
    q4_routes = queries.get_downstream_route_impacts("INC-ANDHERI-FLOOD-01", max_hops=3)
    print("\n--- Query 4: Downstream Route & Transit Impacts ---")
    print(f"  - Total Impacted Corridors: {q4_routes['total_impacted_routes']}")
    for r in q4_routes["impacted_corridors"]:
        print(f"      * [{r['node_type']}] {r['route_id']}: {r['label']} (Impact: {r['impact_type']}, Conf: {r['cumulative_impact_confidence']})")

    # Query 5: Safe Alternative Route Avoiding Flooded Segments
    # Build road network graph for demonstration
    road_net = nx.Graph()
    road_net.add_edge("Bandra_Junction", "SV_Road_South", weight=2.0)
    road_net.add_edge("SV_Road_South", "ROUTE-ANDHERI-SUBWAY", weight=1.5)
    road_net.add_edge("ROUTE-ANDHERI-SUBWAY", "SV_Road_North", weight=1.5)
    road_net.add_edge("SV_Road_North", "Goregaon_Hub", weight=3.0)
    # Bypass via Western Express Highway
    road_net.add_edge("Bandra_Junction", "WEH_Flyover_South", weight=2.5)
    road_net.add_edge("WEH_Flyover_South", "ROUTE-WEH-ALTERNATIVE", weight=2.0)
    road_net.add_edge("ROUTE-WEH-ALTERNATIVE", "Goregaon_Hub", weight=2.5)

    q5_routing = queries.find_safe_alternative_route(
        start_node="Bandra_Junction",
        end_node="Goregaon_Hub",
        blocked_nodes={"ROUTE-ANDHERI-SUBWAY"},
        road_network_graph=road_net
    )
    print("\n--- Query 5: Alternative Route Bypassing Flood Blockage ---")
    print(f"  - Alternative Found: {q5_routing['route_found']}")
    print(f"  - Avoided Blocked Nodes: {q5_routing['avoided_nodes']}")
    print(f"  - Computed Path: {' -> '.join(q5_routing['path_nodes'])}")
    print(f"  - {q5_routing['causation_disclaimer']}")

    # 4. Serialize Outputs
    json_path = os.path.join(CURRENT_DIR, "sample_scenario_graph.json")
    eg.save_to_json(json_path)
    print(f"\n[Export] Saved sample scenario graph JSON to: {json_path}")

    results_dict = {
        "scenario_name": "Andheri Subway Monsoon Inundation",
        "validation_audit": audit,
        "query_1_evidence": q1_evidence,
        "query_2_contradictions": q2_contra,
        "query_3_infrastructure": q3_infra,
        "query_4_downstream_impacts": q4_routes,
        "query_5_safe_routing": q5_routing
    }
    results_json_path = os.path.join(CURRENT_DIR, "graph_scenario_query_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[Export] Saved query results JSON to: {results_json_path}")

    print("\n" + "=" * 75)
    print("PHASE 5 SCENARIO PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_scenario_pipeline()
