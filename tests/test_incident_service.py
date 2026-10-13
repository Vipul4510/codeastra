"""
Unit and Integration Tests for UrbanPulse Incident Normalization & Integration (Phase 7B.2)
========================================================================================
Verifies:
1. Valid incident normalization across agent API outputs, live_events, and dataset clusters.
2. Handling of missing fields without inventing values.
3. Strict validation on malformed coordinates, timestamps, and numerical fields.
4. Strict architectural decoupling:
   - Classification model_confidence != route_impact_confidence.
   - RED_CRITICAL alert or 1.0 confidence does NOT mean road is blocked.
   - Nearby physical assets are spatial proximities, NOT confirmed root causes.
5. Active incident retrieval function (get_active_incidents) without duplicating storage.
6. Backward compatibility: existing agent endpoints (telemetry ingestion, incidents list, location query) remain intact.
"""

import os
import sys
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from incident_service import (
    NormalizedIncident,
    IncidentValidationError,
    CoordinateValidationError,
    TimestampValidationError,
    NumericalFieldValidationError,
    validate_coordinates,
    validate_timestamp,
    validate_numerical_field,
    calculate_route_impact_confidence,
    normalize_incident,
    get_active_incidents,
)
from api_service import app, live_events


class TestIncidentNormalizationValid(unittest.TestCase):
    """Verifies correct parsing of valid incident payloads from different sources."""

    def test_normalize_agent_api_output(self):
        # Shape returned by POST /api/v1/telemetry
        raw_agent_output = {
            "incident_id": "EVT-LIVE-00001",
            "timestamp": "2026-10-13T04:20:00+05:30",
            "incident_location": {
                "locality": "Andheri West",
                "coordinates": {
                    "latitude": 19.11985,
                    "longitude": 72.8465
                },
                "nearby_critical_assets": [
                    {"asset_id": "AST-PUMP-004", "asset_type": "PUMP_STATION", "distance_km": 0.25}
                ]
            },
            "model_evaluation": {
                "predicted_relationship": "NEW_PRIMARY_INCIDENT",
                "confidence": 0.95,
                "note": "First incoming observation in active window"
            },
            "generated_response": {
                "alert_level": "AMBER_WARNING",
                "urgency": "HIGH",
                "root_cause_hypothesis": "Localized runoff accumulation impeding vehicular movement",
                "traffic_impact": {
                    "passability": "RESTRICTED",
                    "speed_drop_pct": 65.0,
                    "delay_minutes": 35.0,
                    "affected_route_id": "ROUTE-004"
                },
                "ml_inference_context": {
                    "predicted_relationship": "NEW_PRIMARY_INCIDENT",
                    "model_confidence": 0.95
                },
                "recommended_actions": [
                    "Dispatch Municipal Ward field inspection team to Andheri West.",
                    "Clear stormwater outfall grates and inspect manhole sumps."
                ]
            }
        }

        norm = normalize_incident(raw_agent_output)

        self.assertEqual(norm.incident_id, "EVT-LIVE-00001")
        self.assertEqual(norm.timestamp, "2026-10-13T04:20:00+05:30")
        self.assertEqual(norm.locality, "Andheri West")
        self.assertEqual(norm.latitude, 19.11985)
        self.assertEqual(norm.longitude, 72.8465)
        self.assertEqual(norm.predicted_relationship, "NEW_PRIMARY_INCIDENT")
        self.assertEqual(norm.model_confidence, 0.95)
        self.assertEqual(norm.alert_level, "AMBER_WARNING")
        self.assertEqual(norm.urgency, "HIGH")
        self.assertEqual(norm.passability, "RESTRICTED")
        self.assertEqual(norm.speed_drop_pct, 65.0)
        self.assertEqual(norm.delay_minutes, 35.0)
        self.assertEqual(norm.affected_route_id, "ROUTE-004")
        self.assertFalse(norm.is_road_blocked)
        self.assertIsNotNone(norm.route_impact_confidence)
        self.assertEqual(len(norm.recommended_actions), 2)
        self.assertEqual(len(norm.nearby_assets), 1)
        self.assertEqual(norm.nearby_assets[0]["asset_id"], "AST-PUMP-004")
        self.assertIn("runoff accumulation", norm.root_cause_hypothesis)
        self.assertEqual(norm.status, "ACTIVE")

    def test_normalize_canonical_dataset_cluster(self):
        # Shape from incident_clusters.csv
        raw_cluster = {
            "incident_id": "INC-000001",
            "incident_type": "ROAD_WATERLOGGING",
            "locality": "Borivali",
            "latitude": 19.231738,
            "longitude": 72.855516,
            "first_seen": "2026-04-17T19:27:32",
            "last_seen": "2026-04-17T20:43:32",
            "severity": "MODERATE",
            "cluster_status": "CANDIDATE",
            "provenance": "SYNTHETIC_GENERATED"
        }

        norm = normalize_incident(raw_cluster)

        self.assertEqual(norm.incident_id, "INC-000001")
        self.assertEqual(norm.timestamp, "2026-04-17T19:27:32")
        self.assertEqual(norm.locality, "Borivali")
        self.assertEqual(norm.latitude, 19.231738)
        self.assertEqual(norm.longitude, 72.855516)
        self.assertEqual(norm.alert_level, "MODERATE")
        self.assertEqual(norm.status, "CANDIDATE")
        self.assertEqual(norm.provenance, "SYNTHETIC_GENERATED")
        # Missing traffic telemetry must remain None
        self.assertIsNone(norm.passability)
        self.assertIsNone(norm.speed_drop_pct)
        self.assertIsNone(norm.delay_minutes)
        self.assertIsNone(norm.affected_route_id)
        self.assertIsNone(norm.route_impact_confidence)
        self.assertFalse(norm.is_road_blocked)


class TestMissingFieldsPreservation(unittest.TestCase):
    """Verifies that missing fields are preserved as None/empty without fabricating values."""

    def test_missing_traffic_metrics_are_none(self):
        minimal_event = {
            "observation_id": "EVT-TEST-001",
            "observation_time": "2026-10-13T05:00:00Z",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "locality": "Kandivali",
            "text": "Water pooling near subway",
            "water_level_cm": 30.0
        }

        norm = normalize_incident(minimal_event)

        # Do NOT invent values
        self.assertIsNone(norm.passability)
        self.assertIsNone(norm.speed_drop_pct)
        self.assertIsNone(norm.delay_minutes)
        self.assertIsNone(norm.affected_route_id)
        self.assertIsNone(norm.predicted_relationship)
        self.assertIsNone(norm.model_confidence)
        self.assertIsNone(norm.route_impact_confidence)
        self.assertFalse(norm.is_road_blocked)
        self.assertEqual(norm.recommended_actions, [])
        self.assertEqual(norm.nearby_assets, [])
        self.assertEqual(norm.evidence.get("water_level_cm"), 30.0)
        self.assertIsNotNone(norm.raw_payload)


class TestMalformedFieldsValidation(unittest.TestCase):
    """Verifies strict rejection of invalid coordinates, timestamps, and numerical bounds."""

    def test_invalid_latitude_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-LAT",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 95.0,  # Out of [-90, 90]
            "longitude": 72.8500
        }
        with self.assertRaises(CoordinateValidationError):
            normalize_incident(bad_payload)

    def test_invalid_longitude_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-LON",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 185.0  # Out of [-180, 180]
        }
        with self.assertRaises(CoordinateValidationError):
            normalize_incident(bad_payload)

    def test_non_finite_coordinates_raise(self):
        bad_payload = {
            "incident_id": "EVT-NAN-COORD",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": float("nan"),
            "longitude": 72.8500
        }
        with self.assertRaises(CoordinateValidationError):
            normalize_incident(bad_payload)

    def test_malformed_timestamp_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-TS",
            "timestamp": "not-a-valid-iso-date",
            "latitude": 19.2000,
            "longitude": 72.8500
        }
        with self.assertRaises(TimestampValidationError):
            normalize_incident(bad_payload)

    def test_missing_timestamp_raises(self):
        bad_payload = {
            "incident_id": "EVT-NO-TS",
            "latitude": 19.2000,
            "longitude": 72.8500
        }
        with self.assertRaises(TimestampValidationError):
            normalize_incident(bad_payload)

    def test_out_of_bounds_speed_drop_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-SPEED",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "speed_drop_pct": 120.0  # Cannot exceed 100%
        }
        with self.assertRaises(NumericalFieldValidationError):
            normalize_incident(bad_payload)

    def test_negative_delay_minutes_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-DELAY",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "delay_minutes": -15.0  # Cannot be negative
        }
        with self.assertRaises(NumericalFieldValidationError):
            normalize_incident(bad_payload)

    def test_out_of_bounds_model_confidence_raises(self):
        bad_payload = {
            "incident_id": "EVT-BAD-CONF",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "model_confidence": 1.5  # Cannot exceed 1.0
        }
        with self.assertRaises(NumericalFieldValidationError):
            normalize_incident(bad_payload)


class TestDecouplingInvariants(unittest.TestCase):
    """
    Verifies critical architectural invariants:
    1. model_confidence is separate from route_impact_confidence.
    2. RED_CRITICAL alert or confidence 1.0 does NOT mean road is blocked.
    3. Nearby assets are spatial proximities, NOT assumed root causes.
    """

    def test_model_confidence_one_does_not_mean_road_blocked(self):
        # Event with 1.0 ML confidence and RED_CRITICAL alert, but passability is PASSABLE
        payload = {
            "incident_id": "EVT-CRITICAL-RUNOFF",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "model_confidence": 1.0,
            "alert_level": "RED_CRITICAL",
            "urgency": "IMMEDIATE",
            "traffic": {
                "road_passability": "PASSABLE",
                "speed_drop_pct": 5.0,
                "traffic_delay_minutes": 2.0
            }
        }

        norm = normalize_incident(payload)

        # ML model confidence is 1.0
        self.assertEqual(norm.model_confidence, 1.0)
        self.assertEqual(norm.alert_level, "RED_CRITICAL")

        # Road must NOT be marked blocked!
        self.assertFalse(norm.is_road_blocked)
        # Route impact confidence must be low, decoupled from model_confidence
        self.assertIsNotNone(norm.route_impact_confidence)
        self.assertLess(norm.route_impact_confidence, 0.40)
        self.assertNotEqual(norm.route_impact_confidence, norm.model_confidence)

    def test_road_blocked_only_on_closed_or_impassable(self):
        # Road is CLOSED
        payload_closed = {
            "incident_id": "EVT-ROAD-CLOSED",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "traffic": {
                "road_passability": "CLOSED",
                "traffic_delay_minutes": 60.0
            }
        }
        norm_closed = normalize_incident(payload_closed)
        self.assertTrue(norm_closed.is_road_blocked)
        self.assertEqual(norm_closed.route_impact_confidence, 0.95)

        # Road is IMPASSABLE
        payload_impassable = {
            "incident_id": "EVT-ROAD-IMPASSABLE",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "traffic": {
                "road_passability": "IMPASSABLE"
            }
        }
        norm_impassable = normalize_incident(payload_impassable)
        self.assertTrue(norm_impassable.is_road_blocked)
        self.assertEqual(norm_impassable.route_impact_confidence, 0.95)

        # Road is RESTRICTED (heavy delay, but not blocked)
        payload_restricted = {
            "incident_id": "EVT-ROAD-RESTRICTED",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "traffic": {
                "road_passability": "RESTRICTED",
                "speed_drop_pct": 60.0,
                "traffic_delay_minutes": 30.0
            }
        }
        norm_restricted = normalize_incident(payload_restricted)
        self.assertFalse(norm_restricted.is_road_blocked)
        self.assertGreaterEqual(norm_restricted.route_impact_confidence, 0.60)

    def test_nearby_asset_not_assumed_root_cause(self):
        payload = {
            "incident_id": "EVT-NEAR-PUMP",
            "timestamp": "2026-10-13T05:00:00+05:30",
            "latitude": 19.2000,
            "longitude": 72.8500,
            "nearby_assets": [
                {"asset_id": "AST-PUMP-009", "distance_km": 0.05, "asset_type": "PUMP_STATION"}
            ],
            "generated_response": {
                "root_cause_hypothesis": "Pumping failure triggered by power feeder disruption"
            }
        }

        norm = normalize_incident(payload)

        # Stored explicitly as spatial proximity under nearby_assets
        self.assertEqual(len(norm.nearby_assets), 1)
        self.assertEqual(norm.nearby_assets[0]["asset_id"], "AST-PUMP-009")

        # Must remain explicitly a hypothesis, not confirmed root cause
        self.assertEqual(norm.root_cause_hypothesis, "Pumping failure triggered by power feeder disruption")


class TestIncidentRetrievalFunction(unittest.TestCase):
    """Verifies get_active_incidents retrieves from existing store without duplicating."""

    def setUp(self):
        live_events.clear()

    def tearDown(self):
        live_events.clear()

    def test_get_active_incidents_from_live_events(self):
        # Populate live_events directly in api_service
        live_events.append({
            "observation_id": "EVT-LIVE-001",
            "observation_time": "2026-10-13T05:10:00+05:30",
            "latitude": 19.1198,
            "longitude": 72.8465,
            "locality": "Andheri West",
            "traffic": {
                "road_passability": "RESTRICTED",
                "speed_drop_pct": 50.0
            }
        })
        live_events.append({
            "observation_id": "EVT-LIVE-002",
            "observation_time": "2026-10-13T05:15:00+05:30",
            "latitude": 19.2300,
            "longitude": 72.8550,
            "locality": "Borivali"
        })

        # Retrieve all active live incidents
        active = get_active_incidents(include_live=True, include_dataset=False)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0].incident_id, "EVT-LIVE-001")
        self.assertEqual(active[1].incident_id, "EVT-LIVE-002")

        # Retrieve filtered by locality
        andheri = get_active_incidents(include_live=True, include_dataset=False, locality="Andheri West")
        self.assertEqual(len(andheri), 1)
        self.assertEqual(andheri[0].locality, "Andheri West")

    def test_get_active_incidents_from_dataset(self):
        # Query active records from incident_clusters.csv
        active_clusters = get_active_incidents(include_live=False, include_dataset=True, locality="Borivali")
        self.assertGreater(len(active_clusters), 0)
        for inc in active_clusters:
            self.assertEqual(inc.locality, "Borivali")
            self.assertIn(inc.status, {"INVESTIGATING", "DETECTED", "CANDIDATE"})


class TestAgentAPIEndpointsIntact(unittest.TestCase):
    """Verifies existing agent endpoints and newly exposed active incidents endpoint."""

    def setUp(self):
        live_events.clear()
        self.client = TestClient(app)

    def tearDown(self):
        live_events.clear()

    def test_ingest_telemetry_and_retrieve_normalized(self):
        # 1. Ingest telemetry report via existing agent endpoint
        payload = {
            "source_modality": "TRAFFIC_FEED",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.11985,
            "longitude": 72.8465,
            "locality": "Andheri West",
            "description_or_text": "Waterlogging causing heavy transit slowdown",
            "water_level_cm": 35.0,
            "traffic": {
                "traffic_speed_kmh": 10.0,
                "speed_drop_pct": 55.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 25.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-004"
            }
        }
        res_post = self.client.post("/api/v1/telemetry", json=payload)
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertIn("incident_id", data_post)
        self.assertIn("generated_response", data_post)

        # 2. Existing GET /api/v1/incidents remains intact
        res_list = self.client.get("/api/v1/incidents")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(res_list.json()["total_active_events"], 1)

        # 3. New GET /api/v1/incidents/active returns normalized structure
        res_active = self.client.get("/api/v1/incidents/active")
        self.assertEqual(res_active.status_code, 200)
        active_list = res_active.json()
        self.assertEqual(len(active_list), 1)

        first_inc = active_list[0]
        self.assertEqual(first_inc["incident_id"], data_post["incident_id"])
        self.assertEqual(first_inc["locality"], "Andheri West")
        self.assertEqual(first_inc["latitude"], 19.11985)
        self.assertEqual(first_inc["longitude"], 72.8465)
        self.assertEqual(first_inc["passability"], "RESTRICTED")
        self.assertEqual(first_inc["speed_drop_pct"], 55.0)
        self.assertEqual(first_inc["delay_minutes"], 25.0)
        self.assertEqual(first_inc["affected_route_id"], "ROUTE-004")
        self.assertFalse(first_inc["is_road_blocked"])
        self.assertIsNotNone(first_inc["route_impact_confidence"])


if __name__ == "__main__":
    unittest.main()
