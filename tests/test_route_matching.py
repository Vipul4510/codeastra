"""
Unit and Integration Tests for Geospatial Incident-to-Route Matching (Phase 7B.3)
================================================================================
Verifies:
1. Incident on route (ON_ROUTE / AFFECTS_ROUTE with CONFIRMED_DISRUPTION).
2. Nearby but unrelated incident (NEAR_ROUTE with NO_ESTABLISHED_DISRUPTION).
3. Incident far from route (NOT_AFFECTED with NO_ESTABLISHED_DISRUPTION).
4. Stale incident (> max_staleness_hours -> STALE).
5. Resolved incident (status == RESOLVED -> RESOLVED).
6. Invalid coordinates (UNKNOWN with INSUFFICIENT_EVIDENCE).
7. Uncertain-flood case (passability unknown -> POSSIBLE_DISRUPTION).
8. Critical incident that is NOT route-affecting (RED_CRITICAL -> NO_ESTABLISHED_DISRUPTION).
9. Opposite-carriageway invariant (does not claim opposite direction impact).
10. Polyline vs straight origin-destination line distance verification.
11. Configurable candidate radius thresholding.
"""

import os
import sys
import unittest
from datetime import datetime, timezone, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from route_matching_service import (
    RouteMatchingService,
    RouteMatchingConfig,
    MatchClassification,
    ImpactStatus,
    compute_point_to_polyline_distance,
    calculate_bearing,
    angular_difference,
    get_route_matching_service
)
from incident_service import NormalizedIncident


# Sample Route Polyline in Mumbai Western Suburbs (Western Express Highway corridor)
# Coordinates ordered [longitude, latitude] as returned by OSRM GeoJSON LineString
SAMPLE_ROUTE_POLYLINE = [
    [72.8567, 19.2307],  # Borivali
    [72.8580, 19.2050],  # Kandivali
    [72.8540, 19.1870],  # Malad
    [72.8490, 19.1660],  # Goregaon
    [72.8480, 19.1350],  # Jogeshwari
    [72.8460, 19.1190]   # Andheri
]


class TestGeospatialPolylineGeometry(unittest.TestCase):
    """Verifies that point-to-polyline distance uses the polyline segments and not a straight line."""

    def test_polyline_vs_straight_line(self):
        # L-shaped route: A (0,0) -> B (0, 0.01) -> C (0.01, 0.01)
        # Point P is placed at (0.0001, 0.01) -> very close to vertex B!
        polyline = [
            [72.8400, 19.1000],  # A
            [72.8400, 19.1200],  # B (approx 2.2 km north)
            [72.8600, 19.1200]   # C (approx 2.1 km east)
        ]
        # P is right next to corner B (distance ~ 15 metres)
        p_lat, p_lon = 19.1201, 72.8401

        dist_m, q_lat, q_lon, seg_idx, _ = compute_point_to_polyline_distance(p_lat, p_lon, polyline)

        # Distance to polyline should be small (< 25 metres)
        self.assertLess(dist_m, 25.0)
        self.assertAlmostEqual(q_lat, 19.1200, places=3)
        self.assertAlmostEqual(q_lon, 72.8400, places=3)

    def test_bearing_and_angular_difference(self):
        # Pure North: 0 deg
        b_north = calculate_bearing(19.1000, 72.8400, 19.1100, 72.8400)
        self.assertAlmostEqual(b_north, 0.0, delta=1.0)

        # Pure South: 180 deg
        b_south = calculate_bearing(19.1100, 72.8400, 19.1000, 72.8400)
        self.assertAlmostEqual(b_south, 180.0, delta=1.0)

        # Angular difference between North and South is 180 deg
        self.assertAlmostEqual(angular_difference(b_north, b_south), 180.0, delta=1.0)
        # Angular difference between North and North-East (45 deg) is 45 deg
        self.assertAlmostEqual(angular_difference(0.0, 45.0), 45.0)


class TestIncidentRouteMatching(unittest.TestCase):
    """Verifies all matching rules and edge cases required for Phase 7B.3."""

    def setUp(self):
        self.service = RouteMatchingService()
        self.ref_time = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)

    def test_incident_on_the_route_with_confirmed_disruption(self):
        # Incident coordinates lie right on the Malad segment (approx 5 metres away)
        # With confirmed CLOSED passability
        incident = {
            "incident_id": "INC-ON-ROUTE-CLOSED",
            "timestamp": "2026-10-13T11:45:00Z",
            "locality": "Malad",
            "latitude": 19.1870,
            "longitude": 72.8540,
            "traffic": {
                "road_passability": "CLOSED",
                "traffic_delay_minutes": 50.0
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-ON-ROUTE-CLOSED")
        self.assertEqual(match.classification, MatchClassification.AFFECTS_ROUTE)
        self.assertEqual(match.impact_status, ImpactStatus.CONFIRMED_DISRUPTION)
        self.assertTrue(match.is_road_blocked)
        self.assertIsNotNone(match.distance_to_route_m)
        self.assertLessEqual(match.distance_to_route_m, 10.0)
        self.assertIn("Roadway closure", match.explanation)
        self.assertIsNotNone(match.nearest_point)

    def test_incident_nearby_but_unrelated(self):
        # Incident is ~120m away from the Western Express Highway in Goregaon,
        # but roadway is explicitly PASSABLE with no delay
        incident = {
            "incident_id": "INC-NEARBY-PASSABLE",
            "timestamp": "2026-10-13T11:30:00Z",
            "locality": "Goregaon",
            "latitude": 19.1668,
            "longitude": 72.8480,
            "traffic": {
                "road_passability": "PASSABLE",
                "speed_drop_pct": 5.0,
                "traffic_delay_minutes": 0.0
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-NEARBY-PASSABLE")
        self.assertEqual(match.classification, MatchClassification.NEAR_ROUTE)
        self.assertEqual(match.impact_status, ImpactStatus.NO_ESTABLISHED_DISRUPTION)
        self.assertFalse(match.is_road_blocked)
        self.assertGreater(match.distance_to_route_m, 30.0)
        self.assertLessEqual(match.distance_to_route_m, 500.0)
        self.assertIn("explicitly reported PASSABLE", match.explanation)

    def test_incident_far_from_route(self):
        # Incident in Versova (lat 19.131, lon 72.815), > 3 km west of the Western Express Highway
        incident = {
            "incident_id": "INC-FAR-VERSOVA",
            "timestamp": "2026-10-13T11:45:00Z",
            "locality": "Versova",
            "latitude": 19.1310,
            "longitude": 72.8150,
            "traffic": {
                "road_passability": "CLOSED"
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-FAR-VERSOVA")
        self.assertEqual(match.classification, MatchClassification.NOT_AFFECTED)
        self.assertEqual(match.impact_status, ImpactStatus.NO_ESTABLISHED_DISRUPTION)
        self.assertFalse(match.is_road_blocked)
        self.assertGreater(match.distance_to_route_m, 500.0)
        self.assertIn("beyond candidate radius", match.explanation)

    def test_incident_stale(self):
        # Incident timestamp is 14 hours old (max_staleness_hours is 6.0)
        incident = {
            "incident_id": "INC-STALE-EVENT",
            "timestamp": "2026-10-12T21:00:00Z",  # 15 hours before ref_time
            "locality": "Andheri",
            "latitude": 19.1190,
            "longitude": 72.8460,
            "traffic": {
                "road_passability": "CLOSED"
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-STALE-EVENT")
        self.assertEqual(match.impact_status, ImpactStatus.STALE)
        self.assertEqual(match.classification, MatchClassification.NOT_AFFECTED)
        self.assertIn("STALE", match.explanation)

    def test_incident_resolved(self):
        # Incident status is RESOLVED
        incident = {
            "incident_id": "INC-RESOLVED-EVENT",
            "timestamp": "2026-10-13T11:00:00Z",
            "status": "RESOLVED",
            "locality": "Malad",
            "latitude": 19.1870,
            "longitude": 72.8540,
            "traffic": {
                "road_passability": "CLOSED"
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-RESOLVED-EVENT")
        self.assertEqual(match.impact_status, ImpactStatus.RESOLVED)
        self.assertEqual(match.classification, MatchClassification.NOT_AFFECTED)
        self.assertIn("RESOLVED", match.explanation)

    def test_incident_invalid_coordinates(self):
        # Latitude out of bounds (95.0)
        incident = {
            "incident_id": "INC-INVALID-COORDS",
            "timestamp": "2026-10-13T11:00:00Z",
            "latitude": 95.0,
            "longitude": 72.8540
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-INVALID-COORDS")
        self.assertEqual(match.classification, MatchClassification.UNKNOWN)
        self.assertEqual(match.impact_status, ImpactStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("invalid", match.explanation.lower())

    def test_uncertain_flood_case(self):
        # "A nearby flood with unknown passability must remain a possible disruption."
        # Waterlogging observed ~150m from route, but passability is unknown (None)
        incident = {
            "incident_id": "INC-UNCERTAIN-FLOOD",
            "timestamp": "2026-10-13T11:30:00Z",
            "locality": "Jogeshwari",
            "latitude": 19.1353,
            "longitude": 72.8495,
            "event_type": "ROAD_WATERLOGGING",
            "water_level_cm": 40.0,
            # No traffic telemetry provided -> passability is unknown
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-UNCERTAIN-FLOOD")
        self.assertEqual(match.impact_status, ImpactStatus.POSSIBLE_DISRUPTION)
        self.assertEqual(match.classification, MatchClassification.NEAR_ROUTE)
        self.assertIn("Passability status is unconfirmed", match.explanation)

    def test_critical_incident_that_is_not_route_affecting(self):
        # "Include a critical incident that is not route-affecting."
        # Severity is RED_CRITICAL with deep water (75cm), but it is at an off-corridor
        # localized pump station with road reported PASSABLE
        incident = {
            "incident_id": "INC-CRITICAL-NOT-AFFECTING",
            "timestamp": "2026-10-13T11:30:00Z",
            "locality": "Kandivali",
            "latitude": 19.2052,
            "longitude": 72.8585,  # ~60m away from highway
            "alert_level": "RED_CRITICAL",
            "water_level_cm": 75.0,
            "traffic": {
                "road_passability": "PASSABLE",
                "speed_drop_pct": 5.0
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-CRITICAL-NOT-AFFECTING")
        self.assertEqual(match.classification, MatchClassification.NEAR_ROUTE)
        self.assertEqual(match.impact_status, ImpactStatus.NO_ESTABLISHED_DISRUPTION)
        self.assertFalse(match.is_road_blocked)
        self.assertIn("explicitly reported PASSABLE", match.explanation)

    def test_opposite_carriageway_does_not_claim_disruption(self):
        # Route travels southward (Borivali -> Andheri, bearing ~ 180°).
        # Incident has direction = "NORTHBOUND" on divided Western Express Highway
        incident = {
            "incident_id": "INC-OPPOSITE-CARRIAGEWAY",
            "timestamp": "2026-10-13T11:30:00Z",
            "locality": "Kandivali",
            "latitude": 19.2050,
            "longitude": 72.8580,  # Physically on corridor
            "direction": "NORTHBOUND",
            "is_bidirectional": False,
            "traffic": {
                "road_passability": "CLOSED"
            }
        }

        match = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            reference_time=self.ref_time
        )

        self.assertEqual(match.incident_id, "INC-OPPOSITE-CARRIAGEWAY")
        self.assertEqual(match.classification, MatchClassification.NEAR_ROUTE)
        self.assertEqual(match.impact_status, ImpactStatus.NO_ESTABLISHED_DISRUPTION)
        self.assertFalse(match.is_road_blocked)
        self.assertIn("opposing carriageway", match.explanation)

    def test_configurable_candidate_radius(self):
        # Incident is ~150m from route
        incident = {
            "incident_id": "INC-RADIUS-TEST",
            "timestamp": "2026-10-13T11:30:00Z",
            "locality": "Malad",
            "latitude": 19.1870,
            "longitude": 72.8554,  # ~147m east of route
            "traffic": {
                "road_passability": "RESTRICTED"
            }
        }

        # 1. With standard 500m radius -> NEAR_ROUTE / AFFECTS_ROUTE
        cfg_500 = RouteMatchingConfig(candidate_radius_m=500.0)
        match_500 = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            config=cfg_500,
            reference_time=self.ref_time
        )
        self.assertNotEqual(match_500.classification, MatchClassification.NOT_AFFECTED)

        # 2. With strict 100m radius -> NOT_AFFECTED
        cfg_100 = RouteMatchingConfig(candidate_radius_m=100.0)
        match_100 = self.service.match_incident(
            incident=incident,
            route_polyline=SAMPLE_ROUTE_POLYLINE,
            config=cfg_100,
            reference_time=self.ref_time
        )
        self.assertEqual(match_100.classification, MatchClassification.NOT_AFFECTED)
        self.assertEqual(match_100.impact_status, ImpactStatus.NO_ESTABLISHED_DISRUPTION)


if __name__ == "__main__":
    unittest.main()
