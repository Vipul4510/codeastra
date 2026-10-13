"""
Unit Tests for UrbanPulse Flood Intelligence & Dependency Engine (Phase 6)
============================================================================
Comprehensive test suite verifying all 10 architectural invariants:
1. Asset connectivity verification before propagating dependencies.
2. Temporal plausibility & physical sequencing (cause must precede effect).
3. Sensor health and data quality penalization.
4. INVARIANT 4: Never infer drain blockage from high water level alone.
5. INVARIANT 5: Never infer power failure caused pump failure without verified supply link.
6. Representation of relationships as POSSIBLE_DEPENDENCY.
7. Evidentiary records (supporting, contradictory, missing, rules used).
8. Strict decoupling of severity level from dependency confidence.
9. Testing alternative explanations and contradictory field observations.
10. Missing evidence results in UNCERTAIN_INVESTIGATING status.
11. Conversion to graph-ready edge schema.
"""

import unittest
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dependency_rules import (
    DependencyType,
    DependencyStatus,
    SeverityLevel,
    FloodRuleConfig,
    DependencyAnalysisResult
)
from flood_dependency_engine import FloodDependencyEngine


class TestFloodDependencyEngine(unittest.TestCase):

    def setUp(self):
        self.config = FloodRuleConfig()
        self.engine = FloodDependencyEngine(config=self.config)

        # Setup test asset topology
        # Feeder-01 powers Pump-01 (Verified)
        # Drain-01 drains to Sump-01 (Verified)
        # Sump-01 adjacent to Road-01 (Verified)
        # Road-01 serves Route-SV (Verified)
        assets = [
            {"asset_id": "AST-FEEDER-01", "asset_type": "POWER_FEEDER", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-FEEDER-UNRELATED", "asset_type": "POWER_FEEDER", "locality": "Borivali", "catchment_id": "CAT-BORI"},
            {"asset_id": "AST-PUMP-01", "asset_type": "PUMP_STATION", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-RAIN-01", "asset_type": "RAIN_GAUGE", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-DRAIN-01", "asset_type": "DRAIN", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-SUMP-01", "asset_type": "SUBWAY_UNDERPASS", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-ROAD-01", "asset_type": "ROAD_SEGMENT", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-ROUTE-SV", "asset_type": "ROUTE", "locality": "Andheri West", "catchment_id": "CAT-ANDH"},
            {"asset_id": "AST-ISOLATED-01", "asset_type": "DRAIN", "locality": "Dahisar", "catchment_id": "CAT-DAHI"}
        ]
        deps = [
            {"from_asset_id": "AST-FEEDER-01", "to_asset_id": "AST-PUMP-01", "relationship_type": "POWERED_BY", "relationship_status": "VERIFIED", "confidence_score": 0.95},
            {"from_asset_id": "AST-RAIN-01", "to_asset_id": "AST-SUMP-01", "relationship_type": "UPSTREAM_OF", "relationship_status": "VERIFIED", "confidence_score": 0.85},
            {"from_asset_id": "AST-DRAIN-01", "to_asset_id": "AST-SUMP-01", "relationship_type": "DRAINS_TO", "relationship_status": "VERIFIED", "confidence_score": 0.90},
            {"from_asset_id": "AST-PUMP-01", "to_asset_id": "AST-SUMP-01", "relationship_type": "DRAINS_TO", "relationship_status": "VERIFIED", "confidence_score": 0.90},
            {"from_asset_id": "AST-SUMP-01", "to_asset_id": "AST-ROAD-01", "relationship_type": "ADJACENT_TO", "relationship_status": "VERIFIED", "confidence_score": 0.88},
            {"from_asset_id": "AST-ROAD-01", "to_asset_id": "AST-ROUTE-SV", "relationship_type": "SERVES_ROAD", "relationship_status": "VERIFIED", "confidence_score": 0.92},
            {"from_asset_id": "AST-SUMP-01", "to_asset_id": "AST-ROUTE-SV", "relationship_type": "IMPACTS_ROUTE", "relationship_status": "VERIFIED", "confidence_score": 0.95}
        ]
        self.engine.load_assets(assets)
        self.engine.load_dependencies(deps)

    def test_asset_connectivity_verification(self):
        """Requirement 1: Verify connectivity before propagating dependencies."""
        # Case A: Direct verified connection
        is_conn, desc, conf = self.engine.verify_asset_connectivity("AST-FEEDER-01", "AST-PUMP-01")
        self.assertTrue(is_conn)
        self.assertIn("DIRECT_EDGE", desc)

        # Case B: Multi-hop path (Drain -> Sump -> Road)
        is_conn_multi, desc_multi, conf_multi = self.engine.verify_asset_connectivity("AST-DRAIN-01", "AST-ROAD-01")
        self.assertTrue(is_conn_multi)
        self.assertIn("MULTI_HOP_PATH", desc_multi)

        # Case C: Unconnected isolated asset in different catchment
        is_conn_none, desc_none, conf_none = self.engine.verify_asset_connectivity("AST-ISOLATED-01", "AST-SUMP-01")
        self.assertFalse(is_conn_none)
        self.assertIsNone(desc_none)

    def test_temporal_plausibility_and_propagation_delay(self):
        """Requirement 2: Ensure timestamps are compatible and sequence is physically plausible."""
        # Case A: Cause precedes effect within 15 minutes -> Plausible
        t_cause = "2026-07-15T14:00:00+05:30"
        t_effect = "2026-07-15T14:15:00+05:30"
        is_plaus, delay, msg = self.engine.check_temporal_plausibility(t_cause, t_effect, max_delay_minutes=60.0)
        self.assertTrue(is_plaus)
        self.assertEqual(delay, 15.0)

        # Case B: Effect occurs 20 minutes BEFORE cause -> Physically Implausible
        t_future_cause = "2026-07-15T14:35:00+05:30"
        t_past_effect = "2026-07-15T14:15:00+05:30"
        is_plaus_bad, delay_bad, msg_bad = self.engine.check_temporal_plausibility(t_future_cause, t_past_effect, max_delay_minutes=60.0)
        self.assertFalse(is_plaus_bad)
        self.assertIn("PHYSICALLY_IMPLAUSIBLE", msg_bad)

        # Case C: Time gap exceeds maximum propagation envelope (180 min vs 60 min)
        t_late_effect = "2026-07-15T17:00:00+05:30"
        is_plaus_late, delay_late, msg_late = self.engine.check_temporal_plausibility(t_cause, t_late_effect, max_delay_minutes=60.0)
        self.assertFalse(is_plaus_late)
        self.assertIn("EXCEEDS_MAX_DELAY", msg_late)

    def test_rainfall_to_water_level_rise_rule(self):
        """Analyze heavy rainfall causing rising water level with alternative hypothesis testing."""
        # Case A: Valid pluvial rainfall flood
        rain_reading = {
            "sensor_id": "AST-RAIN-01",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "rainfall_15min_mm": 12.5,
            "rainfall_1hr_mm": 45.0,
            "data_quality": "VALID",
            "sensor_health": "HEALTHY"
        }
        water_reading = {
            "sensor_id": "AST-SUMP-01",
            "timestamp": "2026-07-15T14:20:00+05:30",
            "water_level_cm": 42.0,
            "water_level_change_cm": 8.5,
            "data_quality": "VALID",
            "sensor_health": "HEALTHY"
        }
        res = self.engine.analyze_rainfall_to_water_rise(rain_reading, water_reading)
        self.assertEqual(res.status, DependencyStatus.POSSIBLE_DEPENDENCY)
        self.assertGreaterEqual(res.confidence_score, 0.70)
        self.assertEqual(res.severity_level, SeverityLevel.HIGH)
        self.assertEqual(len(res.supporting_evidence), 2)

        # Case B: Water rising with NO rainfall -> Refutes pluvial hypothesis and surfaces alternatives
        dry_rain_reading = {
            "sensor_id": "AST-RAIN-01",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "rainfall_15min_mm": 0.0,
            "rainfall_1hr_mm": 0.5,
            "data_quality": "VALID"
        }
        res_dry = self.engine.analyze_rainfall_to_water_rise(dry_rain_reading, water_reading)
        self.assertEqual(res_dry.status, DependencyStatus.REFUTED_CONTRADICTED)
        self.assertGreater(len(res_dry.alternative_hypotheses), 0)
        self.assertIn("High tide marine backflow", res_dry.alternative_hypotheses)

    def test_invariant_4_no_drain_blockage_from_high_water_alone(self):
        """
        CRITICAL INVARIANT 4:
        Do NOT infer a drain blockage from high water level alone!
        """
        drain_reading_normal = {
            "sensor_id": "AST-DRAIN-01",
            "timestamp": "2026-07-15T14:10:00+05:30",
            "blockage_probability": 0.08,  # Low blockage probability
            "drainage_capacity_pct": 88.0,
        }
        sump_reading_flooded = {
            "sensor_id": "AST-SUMP-01",
            "timestamp": "2026-07-15T14:20:00+05:30",
            "water_level_cm": 62.0,  # Extremely high water level!
        }

        # Attempt to infer blockage from high water level alone
        res = self.engine.analyze_drain_blockage_to_accumulation(drain_reading_normal, sump_reading_flooded)

        # MUST NOT BE CONFIRMED OR POSSIBLE DEPENDENCY! Must be UNCERTAIN_INVESTIGATING
        self.assertEqual(res.status, DependencyStatus.UNCERTAIN_INVESTIGATING)
        self.assertIn("INVARIANT ENFORCED", res.explanation)
        self.assertIn("High water level alone is insufficient to infer a drain blockage", res.explanation)
        self.assertGreater(len(res.alternative_hypotheses), 0)
        self.assertIn("Rainfall volume simply exceeded nominal hydraulic design capacity.", res.alternative_hypotheses)

        # Case B: Valid blockage when explicit blockage telemetry exists
        drain_reading_blocked = {
            "sensor_id": "AST-DRAIN-01",
            "timestamp": "2026-07-15T14:10:00+05:30",
            "blockage_probability": 0.78,  # Explicit high blockage indicator!
            "drainage_capacity_pct": 18.0,
        }
        res_blocked = self.engine.analyze_drain_blockage_to_accumulation(drain_reading_blocked, sump_reading_flooded)
        self.assertEqual(res_blocked.status, DependencyStatus.POSSIBLE_DEPENDENCY)
        self.assertGreaterEqual(res_blocked.confidence_score, 0.70)
        self.assertEqual(res_blocked.severity_level, SeverityLevel.CRITICAL)

    def test_invariant_5_no_power_failure_blamed_on_pump_without_supply_link(self):
        """
        CRITICAL INVARIANT 5:
        Do NOT infer that a power failure caused a pump failure unless:
        1. The supply dependency (POWERED_BY) is verified, AND
        2. Supporting power failure evidence exists and temporally precedes pump stoppage.
        """
        # Case A: Power outage occurs on UNRELATED feeder AST-FEEDER-UNRELATED
        unrelated_feeder = {
            "asset_id": "AST-FEEDER-UNRELATED",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "status": "OFFLINE"
        }
        pump_failed = {
            "asset_id": "AST-PUMP-01",
            "timestamp": "2026-07-15T14:10:00+05:30",
            "pump_status": "FAILED"
        }

        res_unrelated = self.engine.analyze_power_to_pump_failure(unrelated_feeder, pump_failed)
        # MUST BE REFUTED: No power dependency exists!
        self.assertEqual(res_unrelated.status, DependencyStatus.REFUTED_CONTRADICTED)
        self.assertIn("INVARIANT ENFORCED: No verified electric supply dependency", res_unrelated.explanation)
        self.assertIn("Mechanical pump impeller jamming (debris/silt)", res_unrelated.alternative_hypotheses)

        # Case B: Outage occurs on the VERIFIED power feeder AST-FEEDER-01
        verified_feeder = {
            "asset_id": "AST-FEEDER-01",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "status": "OFFLINE"
        }
        res_verified = self.engine.analyze_power_to_pump_failure(verified_feeder, pump_failed)
        self.assertEqual(res_verified.status, DependencyStatus.CONFIRMED_DEPENDENCY)
        self.assertGreaterEqual(res_verified.confidence_score, 0.85)
        self.assertEqual(res_verified.severity_level, SeverityLevel.HIGH)

    def test_water_level_to_road_waterlogging_rule(self):
        """Rule 4: Culvert water level surcharge backed up onto surface carriage-way."""
        water_sensor = {
            "sensor_id": "AST-SUMP-01",
            "timestamp": "2026-07-15T14:15:00+05:30",
            "water_level_cm": 48.0
        }
        road_obs = {
            "asset_id": "AST-ROAD-01",
            "observed_time": "2026-07-15T14:30:00+05:30",
            "water_depth_cm": 32.0,
            "road_passability": "RESTRICTED"
        }
        res = self.engine.analyze_water_level_to_road_waterlogging(water_sensor, road_obs)
        self.assertEqual(res.status, DependencyStatus.POSSIBLE_DEPENDENCY)
        self.assertEqual(res.severity_level, SeverityLevel.HIGH)
        self.assertEqual(res.temporal_delay_minutes, 15.0)

    def test_road_waterlogging_to_traffic_slowdown_and_contradiction(self):
        """Rule 5 & Requirement 9: Traffic friction vs contradictory free-flow observations."""
        road_obs = {
            "asset_id": "AST-ROAD-01",
            "observed_time": "2026-07-15T14:30:00+05:30",
            "water_depth_cm": 45.0,
            "road_passability": "CLOSED"
        }
        traffic_disrupted = {
            "route_id": "AST-ROUTE-SV",
            "timestamp": "2026-07-15T14:40:00+05:30",
            "speed_drop_pct": 65.0,
            "congestion_delay_min": 35.0,
            "traffic_status": "CONGESTED"
        }
        res_disrupted = self.engine.analyze_road_waterlogging_to_traffic_slowdown(road_obs, traffic_disrupted)
        self.assertEqual(res_disrupted.status, DependencyStatus.POSSIBLE_DEPENDENCY)
        self.assertEqual(res_disrupted.severity_level, SeverityLevel.CRITICAL)

        # Contradictory observation: Traffic sensor reports normal free-flow
        traffic_free_flow = {
            "route_id": "AST-ROUTE-SV",
            "timestamp": "2026-07-15T14:40:00+05:30",
            "speed_drop_pct": 2.0,
            "congestion_delay_min": 0.0,
            "traffic_status": "FREE_FLOW"
        }
        res_contradicted = self.engine.analyze_road_waterlogging_to_traffic_slowdown(road_obs, traffic_free_flow)
        self.assertEqual(res_contradicted.status, DependencyStatus.REFUTED_CONTRADICTED)
        self.assertGreater(len(res_contradicted.contradictory_evidence), 0)

    def test_underpass_inundation_to_travel_disruption(self):
        """Rule 6: Underpass barrel submerged with 55cm water enforcing closure."""
        underpass_obs = {
            "asset_id": "AST-SUMP-01",
            "observed_time": "2026-07-15T14:30:00+05:30",
            "water_depth_cm": 55.0,
            "road_passability": "CLOSED"
        }
        transit_data = {
            "transit_id": "AST-ROUTE-SV",
            "timestamp": "2026-07-15T14:35:00+05:30",
            "status": "TOTAL_CLOSURE"
        }
        res = self.engine.analyze_underpass_inundation_to_travel_disruption(underpass_obs, transit_data)
        self.assertEqual(res.status, DependencyStatus.POSSIBLE_DEPENDENCY)
        self.assertEqual(res.severity_level, SeverityLevel.CRITICAL)
        self.assertGreaterEqual(res.confidence_score, 0.80)

    def test_severity_decoupled_from_confidence(self):
        """
        CRITICAL INVARIANT 8:
        Severity is assigned strictly separately from dependency confidence.
        """
        # Case A: Degraded/suspect sensor -> Confidence is heavily penalized, but physical severity is CRITICAL
        rain_suspect = {
            "sensor_id": "AST-RAIN-01",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "rainfall_15min_mm": 18.0,
            "rainfall_1hr_mm": 65.0,
            "data_quality": "SUSPECT",  # Penalty
            "sensor_health": "OFFLINE"   # Penalty
        }
        water_critical = {
            "sensor_id": "AST-SUMP-01",
            "timestamp": "2026-07-15T14:20:00+05:30",
            "water_level_cm": 75.0,  # CRITICAL physical level!
            "water_level_change_cm": 15.0,
            "data_quality": "SUSPECT",
            "sensor_health": "DEGRADED"
        }
        res = self.engine.analyze_rainfall_to_water_rise(rain_suspect, water_critical)

        # Severity MUST remain CRITICAL despite low confidence!
        self.assertEqual(res.severity_level, SeverityLevel.CRITICAL)
        self.assertLess(res.confidence_score, 0.50)  # Penalized confidence

    def test_graph_edge_conversion(self):
        """Convert dependency analysis result into validated UrbanPulse evidence graph edge format."""
        verified_feeder = {
            "asset_id": "AST-FEEDER-01",
            "timestamp": "2026-07-15T14:00:00+05:30",
            "status": "OFFLINE"
        }
        pump_failed = {
            "asset_id": "AST-PUMP-01",
            "timestamp": "2026-07-15T14:10:00+05:30",
            "pump_status": "FAILED"
        }
        res = self.engine.analyze_power_to_pump_failure(verified_feeder, pump_failed)
        edge_dict = self.engine.convert_analysis_to_graph_edge(res)

        self.assertIsNotNone(edge_dict)
        self.assertEqual(edge_dict["source"], "AST-FEEDER-01")
        self.assertEqual(edge_dict["target"], "AST-PUMP-01")
        self.assertEqual(edge_dict["relationship_type"], "POSSIBLE_DEPENDENCY")
        self.assertIn("AST-FEEDER-01", edge_dict["evidence_ids"])
        self.assertIn("confidence", edge_dict)
        self.assertIn("explanation", edge_dict)


if __name__ == "__main__":
    unittest.main()
