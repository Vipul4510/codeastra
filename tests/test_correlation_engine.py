"""
Unit Tests for UrbanPulse Rule-Based Incident Correlation Engine (Phase 3)
===========================================================================
Covers all five fundamental relationship classes and core architectural invariants:
1. SAME_INCIDENT: Multimodal corroboration within spatio-temporal envelope.
2. DUPLICATE: Ingestion duplicates, social repost chains (single evidence chain).
3. CONTRADICTORY: Mutually exclusive factual claims at same locality/time.
4. RELATED_BUT_DISTINCT: Causal cascading disruptions across the corridor.
5. UNRELATED: Disparate events violating temporal or spatial bounds.
6. PRINCIPLE 5 SAFEGUARD: Never merge solely due to co-location/co-occurrence.
7. SEVERITY DECOUPLING: Severity evaluated strictly separately from identity & confidence.
8. CLUSTER FORMATION: Connected components with contradiction isolation.
"""

import unittest
import sys
import os

# Ensure project root is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rule_definitions import (
    CorrelationRuleConfig,
    haversine_distance_km,
    compute_token_jaccard_similarity,
    detect_contradictory_text,
    CAUSAL_RELATIONSHIPS
)
from correlation_engine import IncidentCorrelationEngine


class TestIncidentCorrelationEngine(unittest.TestCase):

    def setUp(self):
        self.config = CorrelationRuleConfig()
        self.engine = IncidentCorrelationEngine(config=self.config)

    def test_haversine_distance_calculation(self):
        """Verify Haversine formula on known Mumbai Western Suburbs coordinates."""
        # Andheri Subway (19.1197, 72.8464) to Milan Subway Santa Cruz (19.0838, 72.8412)
        dist = haversine_distance_km(19.1197, 72.8464, 19.0838, 72.8412)
        # Expected distance is ~4.0 km
        self.assertAlmostEqual(dist, 4.02, delta=0.2)

        # Same point distance must be 0.0
        self.assertEqual(haversine_distance_km(19.1197, 72.8464, 19.1197, 72.8464), 0.0)

    def test_same_incident_corroboration(self):
        """
        Requirement 1 & 8: Localized waterlogging citizen report + sensor reading
        within 0.35 km, 12 minutes, matching category -> SAME_INCIDENT.
        """
        event_citizen = {
            "event_id": "EVT-ANDH-01",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "source_reliability": 0.70,
            "observation_time_iso": "2026-07-15T14:30:00+05:30",
            "observation_time_epoch_ms": 1784106000000,
            "latitude": 19.1197,
            "longitude": 72.8464,
            "locality_normalized": "Andheri West",
            "text_raw": "Heavy waterlogging on SV road near Andheri station depth 35cm",
            "text_cleaned": "heavy waterlogging sv road andheri station depth 35cm",
        }

        event_sensor = {
            "event_id": "EVT-ANDH-02",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "source_channel": "SENSOR",
            "source_modality": "SENSOR_TELEMETRY",
            "source_reliability": 0.95,
            "observation_time_iso": "2026-07-15T14:42:00+05:30",
            "observation_time_epoch_ms": 1784106720000,  # 12 minutes later
            "latitude": 19.1215,
            "longitude": 72.8480,  # ~0.26 km away
            "locality_normalized": "Andheri West",
            "text_raw": "Road water level sensor reading 38cm depth waterlogging alert",
            "text_cleaned": "road water level sensor reading 38cm depth waterlogging alert",
        }

        decision = self.engine.evaluate_observation_pair(event_citizen, event_sensor)

        self.assertEqual(decision["relation_label"], "SAME_INCIDENT")
        self.assertGreaterEqual(decision["confidence_score"], 0.65)
        self.assertEqual(decision["rule_id"], "RULE_SAME_INCIDENT_CORROBORATION")
        self.assertEqual(decision["evidence_chain_count"], 2)
        self.assertIn("Corroborating observations of the same localized incident", decision["explanation"])
        self.assertLess(decision["geo_distance_km"], 0.85)
        self.assertEqual(decision["time_diff_minutes"], 12.0)

    def test_exact_duplicate_and_repost_evidence_chain(self):
        """
        Requirement 6: Reposts / forwarded copies treated as ONE evidence chain.
        """
        # Case A: Viral social media repost chain
        event_original = {
            "event_id": "EVT-TWEET-01",
            "raw_id": "POST-1001",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "source_channel": "TWITTER",
            "source_modality": "SOCIAL_MEDIA",
            "source_reliability": 0.60,
            "observation_time_iso": "2026-07-15T15:00:00+05:30",
            "observation_time_epoch_ms": 1784107800000,
            "latitude": 19.1197,
            "longitude": 72.8464,
            "locality_normalized": "Andheri West",
            "text_raw": "Andheri subway completely blocked due to 50cm flooding!",
            "text_cleaned": "andheri subway completely blocked 50cm flooding",
        }

        event_repost = {
            "event_id": "EVT-TWEET-02",
            "raw_id": "POST-1002",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "source_channel": "TWITTER",
            "source_modality": "SOCIAL_MEDIA",
            "source_reliability": 0.50,
            "observation_time_iso": "2026-07-15T15:30:00+05:30",
            "observation_time_epoch_ms": 1784109600000,
            "latitude": 19.1197,
            "longitude": 72.8464,
            "locality_normalized": "Andheri West",
            "text_raw": "RT @citizen: Andheri subway completely blocked due to 50cm flooding!",
            "text_cleaned": "andheri subway completely blocked 50cm flooding",
            "is_repost_or_copy": True,
            "repost_parent_id": "POST-1001",
            "repost_similarity": 0.96,
        }

        decision_repost = self.engine.evaluate_observation_pair(event_original, event_repost)
        self.assertEqual(decision_repost["relation_label"], "DUPLICATE")
        self.assertEqual(decision_repost["rule_id"], "RULE_EXACT_DUPLICATE_OR_REPOST_CHAIN")
        # Evidence chain count MUST be 1 for copied posts!
        self.assertEqual(decision_repost["evidence_chain_count"], 1)
        self.assertIn("one evidence chain", decision_repost["explanation"])

        # Case B: Ingestion fingerprint exact duplicate
        event_dupe = {
            "event_id": "EVT-DUPE-01",
            "duplicate_of_id": "EVT-TWEET-01",
            "duplicate_type": "EXACT_CONTENT_FINGERPRINT",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "observation_time_epoch_ms": 1784107800000,
            "locality_normalized": "Andheri West",
            "latitude": 19.1197,
            "longitude": 72.8464,
        }
        decision_fingerprint = self.engine.evaluate_observation_pair(event_original, event_dupe)
        self.assertEqual(decision_fingerprint["relation_label"], "DUPLICATE")
        self.assertEqual(decision_fingerprint["evidence_chain_count"], 1)

    def test_factual_contradiction_detection(self):
        """
        Requirement 7: Preserve contradictions and missing information.
        Do not convert unknown information into a positive match.
        """
        event_clear = {
            "event_id": "EVT-MILAN-01",
            "canonical_event_type": "GENERAL_INCIDENT",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "source_reliability": 0.70,
            "observation_time_iso": "2026-07-15T11:00:00+05:30",
            "observation_time_epoch_ms": 1784093400000,
            "latitude": 19.0838,
            "longitude": 72.8412,
            "locality_normalized": "Santa Cruz",
            "text_raw": "Milan subway is completely clear and open for vehicles, normal traffic",
            "text_cleaned": "milan subway clear open vehicles normal traffic",
        }

        event_flooded = {
            "event_id": "EVT-MILAN-02",
            "canonical_event_type": "SUBWAY_INUNDATION",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "source_reliability": 0.75,
            "observation_time_iso": "2026-07-15T11:10:00+05:30",
            "observation_time_epoch_ms": 1784094000000,  # 10 min later
            "latitude": 19.0840,
            "longitude": 72.8415,
            "locality_normalized": "Santa Cruz",
            "text_raw": "Milan subway is submerged and closed, avoid completely!",
            "text_cleaned": "milan subway submerged closed avoid",
        }

        decision = self.engine.evaluate_observation_pair(event_clear, event_flooded)
        self.assertEqual(decision["relation_label"], "CONTRADICTORY")
        self.assertEqual(decision["rule_id"], "RULE_CONTRADICTORY_ASSERTION")
        self.assertIn("Contradictory observations", decision["explanation"])
        self.assertIn("Conflict:", decision["explanation"])

    def test_related_but_distinct_causal_cascade(self):
        """
        Requirement: Cascading corridor events (e.g. Heavy rainfall causing downstream
        flooding 2.0 km away within 35 minutes) -> RELATED_BUT_DISTINCT.
        """
        event_rainfall = {
            "event_id": "EVT-RAIN-01",
            "canonical_event_type": "HEAVY_RAINFALL",
            "source_channel": "WEATHER_TELEMETRY",
            "source_modality": "SENSOR_TELEMETRY",
            "source_reliability": 0.90,
            "observation_time_iso": "2026-07-15T12:00:00+05:30",
            "observation_time_epoch_ms": 1784097000000,
            "latitude": 19.1663,
            "longitude": 72.8526,  # Goregaon
            "locality_normalized": "Goregaon",
            "text_raw": "Automatic weather station alert: intense downpour 65mm/hr",
            "text_cleaned": "automatic weather station intense downpour 65mm hr",
        }

        event_flood = {
            "event_id": "EVT-FLOOD-01",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "source_reliability": 0.70,
            "observation_time_iso": "2026-07-15T12:35:00+05:30",
            "observation_time_epoch_ms": 1784099100000,  # 35 min later
            "latitude": 19.1860,
            "longitude": 72.8485,  # Malad (~2.2 km away)
            "locality_normalized": "Malad",
            "text_raw": "SV Road near Malad subway waterlogging rising quickly",
            "text_cleaned": "sv road malad subway waterlogging rising quickly",
        }

        decision = self.engine.evaluate_observation_pair(event_rainfall, event_flood)
        self.assertEqual(decision["relation_label"], "RELATED_BUT_DISTINCT")
        self.assertEqual(decision["rule_id"], "RULE_RELATED_DISTINCT_CORRIDOR_CASCADE")
        self.assertIn("Cascading or corridor-related incident", decision["explanation"])
        self.assertGreater(decision["geo_distance_km"], 0.85)  # Outside strict same-incident envelope

    def test_principle_5_co_located_different_events_safeguard(self):
        """
        Requirement 5 (CRITICAL PRINCIPLE 5):
        Never merge observations solely because they are geographically close or occur around the same time.
        Tree fall vs Power failure 150 meters apart at the same time without textual or causal link.
        """
        event_tree = {
            "event_id": "EVT-TREE-01",
            "canonical_event_type": "TREE_FALL",
            "source_channel": "CITIZEN_REPORT",
            "source_modality": "CITIZEN_REPORT",
            "source_reliability": 0.80,
            "observation_time_iso": "2026-07-15T16:00:00+05:30",
            "observation_time_epoch_ms": 1784111400000,
            "latitude": 19.0596,
            "longitude": 72.8295,  # Bandra West
            "locality_normalized": "Bandra West",
            "text_raw": "Banyan tree branch fell on parked car near Hill Road bakery",
            "text_cleaned": "banyan tree branch fell parked car hill road bakery",
        }

        # Event B is close (100m away, same minute), but represents unrelated localized civic issue
        event_manhole = {
            "event_id": "EVT-MANHOLE-01",
            "canonical_event_type": "MANHOLE_OVERFLOW",
            "source_channel": "MUNICIPAL_TELEMETRY",
            "source_modality": "SENSOR_TELEMETRY",
            "source_reliability": 0.85,
            "observation_time_iso": "2026-07-15T16:05:00+05:30",
            "observation_time_epoch_ms": 1784111700000,  # 5 min later
            "latitude": 19.0602,
            "longitude": 72.8300,  # ~80 meters away
            "locality_normalized": "Bandra West",
            "text_raw": "Stormwater drain inspection cover displaced and gushing water",
            "text_cleaned": "stormwater drain inspection cover displaced gushing water",
        }

        decision = self.engine.evaluate_observation_pair(event_tree, event_manhole)
        # MUST NOT BE SAME_INCIDENT despite being 80m and 5 min apart!
        self.assertNotEqual(decision["relation_label"], "SAME_INCIDENT")
        self.assertEqual(decision["relation_label"], "RELATED_BUT_DISTINCT")
        self.assertEqual(decision["rule_id"], "RULE_CO_LOCATED_INDEPENDENT_EVENTS")
        self.assertIn("Principle 5 Safeguard", decision["explanation"])

    def test_unrelated_events_hard_cutoffs(self):
        """
        Requirement 4 & 5: Hard spatial and temporal cutoffs.
        """
        event_bandra = {
            "event_id": "EVT-BAN-01",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "observation_time_epoch_ms": 1784100000000,
            "latitude": 19.0596,
            "longitude": 72.8295,  # Bandra
            "locality_normalized": "Bandra West",
            "text_raw": "Water on Linking Road Bandra",
        }

        # Far away spatially (> 5 km: Dahisar is ~18 km away)
        event_dahisar = {
            "event_id": "EVT-DAH-01",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "observation_time_epoch_ms": 1784100600000,  # 10 min later
            "latitude": 19.2500,
            "longitude": 72.8590,  # Dahisar
            "locality_normalized": "Dahisar",
            "text_raw": "Water on Dahisar checknaka",
        }

        decision_dist = self.engine.evaluate_observation_pair(event_bandra, event_dahisar)
        self.assertEqual(decision_dist["relation_label"], "UNRELATED")
        self.assertEqual(decision_dist["rule_id"], "RULE_HARD_SPATIAL_TEMPORAL_CUTOFF")
        self.assertIn("Exceeds maximum candidate geographic distance", decision_dist["explanation"])

        # Exceeds temporal window (> 120 min: 180 min later)
        event_bandra_late = {
            "event_id": "EVT-BAN-02",
            "canonical_event_type": "ROAD_WATERLOGGING",
            "observation_time_epoch_ms": 1784100000000 + (180 * 60 * 1000),  # 3 hours later
            "latitude": 19.0596,
            "longitude": 72.8295,
            "locality_normalized": "Bandra West",
            "text_raw": "Water on Linking Road Bandra evening report",
        }

        decision_time = self.engine.evaluate_observation_pair(event_bandra, event_bandra_late)
        self.assertEqual(decision_time["relation_label"], "UNRELATED")
        self.assertEqual(decision_time["rule_id"], "RULE_HARD_SPATIAL_TEMPORAL_CUTOFF")
        self.assertIn("Exceeds maximum candidate temporal window", decision_time["explanation"])

    def test_candidate_clustering_and_contradiction_isolation(self):
        """
        Requirement 9: Build candidate incident clusters while allowing separate incidents
        to remain separate and preserving contradictions without false merges.
        """
        events = [
            # Member 1 & 2 of Cluster A (Andheri Flood)
            {
                "event_id": "EVT-C1",
                "canonical_event_type": "SUBWAY_INUNDATION",
                "source_modality": "CITIZEN_REPORT",
                "source_channel": "CITIZEN",
                "source_reliability": 0.8,
                "latitude": 19.1197,
                "longitude": 72.8464,
                "locality_normalized": "Andheri West",
                "observation_time_iso": "2026-07-15T14:00:00+05:30",
                "observation_time_epoch_ms": 1784104200000,
                "text_raw": "Andheri subway flooded 55cm closed for traffic",
            },
            {
                "event_id": "EVT-C2",
                "canonical_event_type": "SUBWAY_INUNDATION",
                "source_modality": "SENSOR_TELEMETRY",
                "source_channel": "SENSOR",
                "source_reliability": 0.95,
                "latitude": 19.1200,
                "longitude": 72.8465,
                "locality_normalized": "Andheri West",
                "observation_time_iso": "2026-07-15T14:10:00+05:30",
                "observation_time_epoch_ms": 1784104800000,
                "text_raw": "Subway water level 60cm pump active",
            },
            # Contradicting Claim
            {
                "event_id": "EVT-C3-CONFLICT",
                "canonical_event_type": "GENERAL_INCIDENT",
                "source_modality": "CITIZEN_REPORT",
                "source_channel": "CITIZEN",
                "source_reliability": 0.5,
                "latitude": 19.1198,
                "longitude": 72.8464,
                "locality_normalized": "Andheri West",
                "observation_time_iso": "2026-07-15T14:05:00+05:30",
                "observation_time_epoch_ms": 1784104500000,
                "text_raw": "Andheri subway is clear open no waterlogging smooth flow",
            },
            # Distinct Independent Incident (Borivali Tree Fall)
            {
                "event_id": "EVT-BORIVALI",
                "canonical_event_type": "TREE_FALL",
                "source_modality": "CITIZEN_REPORT",
                "source_channel": "CITIZEN",
                "source_reliability": 0.7,
                "latitude": 19.2300,
                "longitude": 72.8567,
                "locality_normalized": "Borivali West",
                "observation_time_iso": "2026-07-15T14:15:00+05:30",
                "observation_time_epoch_ms": 1784105100000,
                "text_raw": "Tree fell on Link Road near Borivali station",
            }
        ]

        # Evaluate pairwise
        pairwise = [
            self.engine.evaluate_observation_pair(events[0], events[1]),  # C1 & C2: SAME_INCIDENT
            self.engine.evaluate_observation_pair(events[0], events[2]),  # C1 & C3: CONTRADICTORY
            self.engine.evaluate_observation_pair(events[0], events[3]),  # C1 & BORIVALI: UNRELATED
        ]

        clusters = self.engine.build_candidate_clusters(events, pairwise)

        # There should be at least 3 clusters (Andheri group, Conflicted singleton, Borivali singleton)
        self.assertGreaterEqual(len(clusters), 3)

        # Find the cluster containing EVT-C1
        c1_cluster = next(c for c in clusters if "EVT-C1" in c["member_event_ids"])
        self.assertIn("EVT-C2", c1_cluster["member_event_ids"])
        # The conflicting claim EVT-C3 MUST NOT be merged into the cluster!
        self.assertNotIn("EVT-C3-CONFLICT", c1_cluster["member_event_ids"])
        # Contradiction flag must be set
        self.assertTrue(c1_cluster["has_contradictions"])
        self.assertGreaterEqual(c1_cluster["active_conflicts_count"], 1)

    def test_severity_assessment_independent_from_identity_and_confidence(self):
        """
        Requirement 10: Keep severity assessment separate from incident identity
        and correlation confidence.
        """
        # Case A: Low Severity, but High Correlation Confidence
        low_sev_events = [
            {
                "event_id": "EVT-LS-1",
                "canonical_event_type": "ROAD_WATERLOGGING",
                "source_reliability": 0.95,
                "text_raw": "Minor puddle on footpath 5cm depth waterlogging",
            },
            {
                "event_id": "EVT-LS-2",
                "canonical_event_type": "ROAD_WATERLOGGING",
                "source_reliability": 0.95,
                "text_raw": "Footpath puddle depth 5cm clear drain",
            }
        ]
        sev_low = self.engine.evaluate_incident_severity(low_sev_events, "ROAD_WATERLOGGING")
        self.assertIn(sev_low["level"], ("LOW", "MODERATE"))

        # Case B: Critical Severity, independent of source count
        crit_event = [{
            "event_id": "EVT-CRIT-1",
            "canonical_event_type": "SUBWAY_INUNDATION",
            "text_raw": "Subway completely flooded depth 65cm impassable closed",
        }]
        sev_crit = self.engine.evaluate_incident_severity(crit_event, "SUBWAY_INUNDATION")
        self.assertEqual(sev_crit["level"], "CRITICAL")
        self.assertIn("65.0cm", sev_crit["rationale"])


if __name__ == "__main__":
    unittest.main()
