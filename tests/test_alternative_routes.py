"""
UrbanPulse Phase 7B.5 — Alternative Route Evaluation & Final Backend Validation
================================================================================
Comprehensive test suite validating:
1. Original route affected by a confirmed closure -> Viable alternative corridor recommended.
2. Nearby incident without established route impact -> Primary corridor remains optimal; no blind delay.
3. Alternative route also affected -> All routes blocked; explicit rejection; no false safety claims.
4. No available alternative -> Single corridor blocked; explicit NO_ALTERNATIVES_FOUND with None recommended ID.
5. OSRM unavailable & timeout -> 502/504 returned; non-fabrication of synthetic geometry.
6. Stale and resolved incidents -> Ignored during route evaluation; corridor remains clear.
7. Missing delay information -> Documented nominal advisory estimate; baseline preserved separately.
8. Multiple journeys with different alert outcomes -> Corridor-isolated alert decisions.
9. Malformed requests & coordinate validation -> 400/422 with precise validation details.
10. Dedicated POST /api/routes/alternatives/evaluate endpoint -> Full candidate breakdown & ranking.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from route_service import (
    OSRMRouteService,
    Coordinate,
    GeoJSONLineString,
    DistanceBreakdown,
    DurationBreakdown,
    RouteSourceInfo,
    RouteLeg,
    RoutePlanResponse,
    set_route_service,
    OSRMServiceError,
    OSRMTimeoutError
)
from api_service import app, live_events


# ------------------------------------------------------------------------------
# Mock Corridors: SV Road (Primary) vs Western Express Highway (Alternative)
# ------------------------------------------------------------------------------
PRIMARY_SV_ROAD = RoutePlanResponse(
    origin=Coordinate(latitude=19.2307, longitude=72.8567),
    destination=Coordinate(latitude=19.1190, longitude=72.8460),
    geometry=GeoJSONLineString(
        type="LineString",
        coordinates=[
            [72.8567, 19.2307],  # Borivali
            [72.8420, 19.2050],  # Kandivali West (SV Road)
            [72.8410, 19.1870],  # Malad West (SV Road)
            [72.8460, 19.1190]   # Andheri West
        ]
    ),
    distance=DistanceBreakdown(meters=14500.0, kilometers=14.5),
    duration=DurationBreakdown(
        seconds=1200.0,
        minutes=20.0,
        traffic_aware=False,
        note="OSRM baseline free-flow duration"
    ),
    source=RouteSourceInfo(
        provider="OSRM",
        profile="driving",
        base_url="http://mock-osrm",
        retrieval_timestamp="2026-10-13T06:00:00Z"
    ),
    legs=[
        RouteLeg(
            distance_meters=14500.0,
            duration_seconds=1200.0,
            summary="Swami Vivekanand Road (SV Road)",
            steps=[]
        )
    ]
)

ALTERNATIVE_WEH = RoutePlanResponse(
    origin=Coordinate(latitude=19.2307, longitude=72.8567),
    destination=Coordinate(latitude=19.1190, longitude=72.8460),
    geometry=GeoJSONLineString(
        type="LineString",
        coordinates=[
            [72.8567, 19.2307],  # Borivali
            [72.8620, 19.2050],  # Kandivali East (WEH)
            [72.8610, 19.1870],  # Malad East (WEH)
            [72.8590, 19.1190]   # Andheri East (WEH)
        ]
    ),
    distance=DistanceBreakdown(meters=16200.0, kilometers=16.2),
    duration=DurationBreakdown(
        seconds=1080.0,
        minutes=18.0,
        traffic_aware=False,
        note="OSRM baseline free-flow duration"
    ),
    source=RouteSourceInfo(
        provider="OSRM",
        profile="driving",
        base_url="http://mock-osrm",
        retrieval_timestamp="2026-10-13T06:00:00Z"
    ),
    legs=[
        RouteLeg(
            distance_meters=16200.0,
            duration_seconds=1080.0,
            summary="Western Express Highway (WEH)",
            steps=[]
        )
    ]
)

CLEAR_POWAI_BKC_ROUTE = RoutePlanResponse(
    origin=Coordinate(latitude=19.1176, longitude=72.9060),
    destination=Coordinate(latitude=19.0657, longitude=72.8687),
    geometry=GeoJSONLineString(
        type="LineString",
        coordinates=[
            [72.9060, 19.1176],  # Powai
            [72.8900, 19.0900],  # Kurla
            [72.8687, 19.0657]   # BKC
        ]
    ),
    distance=DistanceBreakdown(meters=9800.0, kilometers=9.8),
    duration=DurationBreakdown(
        seconds=900.0,
        minutes=15.0,
        traffic_aware=False,
        note="OSRM baseline free-flow duration"
    ),
    source=RouteSourceInfo(
        provider="OSRM",
        profile="driving",
        base_url="http://mock-osrm",
        retrieval_timestamp="2026-10-13T06:00:00Z"
    ),
    legs=[
        RouteLeg(
            distance_meters=9800.0,
            duration_seconds=900.0,
            summary="JVLR - SCLR Corridor",
            steps=[]
        )
    ]
)


class TestAlternativeRouteEvaluation(unittest.TestCase):
    """End-to-end and unit validation for Phase 7B.5 Alternative Route Evaluation."""

    def setUp(self):
        live_events.clear()
        self.mock_route_svc = MagicMock()
        # Default mock returns both SV Road (Primary) and Western Express Highway (Alternative)
        self.mock_route_svc.plan_routes.return_value = [PRIMARY_SV_ROAD, ALTERNATIVE_WEH]
        self.mock_route_svc.plan_route.return_value = PRIMARY_SV_ROAD
        set_route_service(self.mock_route_svc)
        self.client = TestClient(app)

    def tearDown(self):
        live_events.clear()
        set_route_service(None)

    # --------------------------------------------------------------------------
    # 1. Original route affected by a confirmed closure
    # --------------------------------------------------------------------------
    def test_original_route_blocked_alternative_recommended(self):
        # Confirmed closure right on SV Road in Kandivali West
        live_events.append({
            "observation_id": "INC-SV-ROAD-CLOSURE",
            "observation_time": "2026-10-13T05:55:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "CLOSED",
                "traffic_delay_minutes": 50.0,
                "affected_route_id": "SV-ROAD"
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        alt = data["alternative_routes"]

        # Status must be ALTERNATIVE_RECOMMENDED
        self.assertEqual(alt["status"], "ALTERNATIVE_RECOMMENDED")
        self.assertTrue(alt["alternatives_available"])
        self.assertEqual(alt["recommended_candidate_id"], "candidate-1")
        self.assertIn("Western Express Highway", alt["recommended_route_summary"])

        # Candidate 0 (SV Road) rejected due to closure
        c0 = alt["candidates"][0]
        self.assertEqual(c0["candidate_id"], "candidate-0")
        self.assertTrue(c0["is_road_blocked"])
        self.assertEqual(c0["recommendation_status"], "REJECTED_BLOCKED")
        self.assertIn("rejected", c0["selection_rationale"].lower())
        self.assertIn("closure", c0["selection_rationale"].lower())

        # Candidate 1 (WEH) selected as PREFERRED
        c1 = alt["candidates"][1]
        self.assertEqual(c1["candidate_id"], "candidate-1")
        self.assertFalse(c1["is_road_blocked"])
        self.assertEqual(c1["recommendation_status"], "PREFERRED")
        self.assertIn("viable", c1["selection_rationale"].lower())
        self.assertEqual(c1["delay"]["baseline_duration_minutes"], 18.0)
        self.assertEqual(c1["delay"]["additional_delay_minutes"], 0.0)

        # Traveler alert indicates detour
        self.assertEqual(data["alert"]["alert_level"], "RED")
        self.assertEqual(data["alert"]["action"], "AVOID_ROUTE")
        self.assertIn("Western Express Highway", data["alert"]["message"])

    # --------------------------------------------------------------------------
    # 2. Nearby incident without established route impact
    # --------------------------------------------------------------------------
    def test_nearby_incident_without_established_route_impact(self):
        # Incident 200m from SV Road but road is PASSABLE
        live_events.append({
            "observation_id": "INC-PASSABLE-NEARBY",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8438,  # ~180m away from SV Road polyline
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "PASSABLE",
                "traffic_delay_minutes": 35.0
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        alt = data["alternative_routes"]

        # Primary corridor is optimal; no detour required
        self.assertEqual(alt["status"], "PRIMARY_ROUTE_OPTIMAL")
        self.assertEqual(alt["recommended_candidate_id"], "candidate-0")

        # Delay must NOT be blindly added to primary route
        self.assertEqual(data["delay"]["additional_delay_minutes"], 0.0)
        self.assertEqual(data["delay"]["total_estimated_duration_minutes"], 20.0)
        self.assertEqual(len(data["impacting_incidents"]), 0)
        self.assertEqual(len(data["nearby_unrelated_incidents"]), 1)

        # Primary candidate is PREFERRED
        c0 = alt["candidates"][0]
        self.assertEqual(c0["recommendation_status"], "PREFERRED")
        self.assertFalse(c0["is_road_blocked"])

    # --------------------------------------------------------------------------
    # 3. Alternative route also affected -> All routes blocked
    # --------------------------------------------------------------------------
    def test_alternative_route_also_affected_all_routes_blocked(self):
        # Closure on SV Road (Candidate 0)
        live_events.append({
            "observation_id": "INC-SV-ROAD-BLOCKED",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "ACTIVE",
            "traffic": {"road_passability": "CLOSED"}
        })
        # AND Closure on Western Express Highway (Candidate 1)
        live_events.append({
            "observation_id": "INC-WEH-BLOCKED",
            "observation_time": "2026-10-13T05:52:00+05:30",
            "locality": "Kandivali East",
            "latitude": 19.2050,
            "longitude": 72.8620,
            "status": "ACTIVE",
            "traffic": {"road_passability": "IMPASSABLE"}
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        alt = data["alternative_routes"]

        # Status must be ALL_ROUTES_BLOCKED
        self.assertEqual(alt["status"], "ALL_ROUTES_BLOCKED")
        self.assertFalse(alt["alternatives_available"])
        self.assertIsNone(alt["recommended_candidate_id"])
        self.assertIsNone(alt["recommended_route_summary"])

        # All candidates must be REJECTED_BLOCKED
        for c in alt["candidates"]:
            self.assertEqual(c["recommendation_status"], "REJECTED_BLOCKED")
            self.assertTrue(c["is_road_blocked"])

        # Explicit warning in selection summary
        self.assertIn("impassable", alt["selection_summary"].lower())
        self.assertIn("no viable alternative", alt["selection_summary"].lower())

        # Top alert is RED / AVOID_ROUTE
        self.assertEqual(data["alert"]["alert_level"], "RED")
        self.assertEqual(data["alert"]["action"], "AVOID_ROUTE")
        self.assertIn("All Routes Blocked", data["alert"]["title"])

    # --------------------------------------------------------------------------
    # 4. No available alternative (single candidate returned, which is blocked)
    # --------------------------------------------------------------------------
    def test_no_available_alternative_when_single_route_blocked(self):
        # Configure mock to return ONLY 1 candidate route
        self.mock_route_svc.plan_routes.return_value = [PRIMARY_SV_ROAD]
        self.mock_route_svc.plan_route.return_value = PRIMARY_SV_ROAD

        # Add road closure to that single corridor
        live_events.append({
            "observation_id": "INC-SOLE-CORRIDOR-CLOSED",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "ACTIVE",
            "traffic": {"road_passability": "CLOSED"}
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        alt = data["alternative_routes"]

        # Status must be NO_ALTERNATIVES_FOUND
        self.assertEqual(alt["status"], "NO_ALTERNATIVES_FOUND")
        self.assertFalse(alt["alternatives_available"])
        self.assertIsNone(alt["recommended_candidate_id"])
        self.assertEqual(alt["total_candidates"], 1)

        c0 = alt["candidates"][0]
        self.assertEqual(c0["recommendation_status"], "REJECTED_BLOCKED")
        self.assertIn("No alternative corridors were generated", alt["selection_summary"])

    # --------------------------------------------------------------------------
    # 5. OSRM unavailable and timeout -> Non-fabrication (502 / 504)
    # --------------------------------------------------------------------------
    def test_osrm_service_unavailable_returns_502(self):
        self.mock_route_svc.plan_routes.side_effect = OSRMServiceError("OSRM connection refused on port 5000")

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("OSRM", resp.json()["detail"])

    def test_osrm_timeout_returns_504(self):
        self.mock_route_svc.plan_routes.side_effect = OSRMTimeoutError("OSRM query timed out after 10.0s")

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 504)
        self.assertIn("timed out", resp.json()["detail"])

    # --------------------------------------------------------------------------
    # 6. Stale and resolved incidents -> Ignored
    # --------------------------------------------------------------------------
    def test_stale_and_resolved_incidents_ignored(self):
        # Incident on SV Road marked RESOLVED
        live_events.append({
            "observation_id": "INC-RESOLVED-FLOOD",
            "observation_time": "2026-10-13T03:00:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "RESOLVED",
            "traffic": {
                "road_passability": "CLOSED",
                "traffic_delay_minutes": 60.0
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        # Resolved incident must NOT impact corridor
        self.assertEqual(len(data["impacting_incidents"]), 0)
        self.assertFalse(data["evaluation_summary"]["is_route_blocked"])
        self.assertEqual(data["alternative_routes"]["status"], "PRIMARY_ROUTE_OPTIMAL")
        self.assertEqual(data["alert"]["alert_level"], "GREEN")

    # --------------------------------------------------------------------------
    # 7. Missing delay information -> Documented nominal estimate
    # --------------------------------------------------------------------------
    def test_missing_delay_information_uses_documented_estimate(self):
        # Restricted road on SV Road with delay_minutes = None
        live_events.append({
            "observation_id": "INC-RESTRICTED-NO-DELAY",
            "observation_time": "2026-10-13T05:55:00+05:30",
            "locality": "Malad West",
            "latitude": 19.1870,
            "longitude": 72.8410,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "RESTRICTED",
                "traffic_delay_minutes": None  # Missing reported delay
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        c0 = data["alternative_routes"]["candidates"][0]

        # Delay breakdown must NOT invent arbitrary 45m; uses documented 5.0m advisory estimate
        self.assertEqual(c0["delay"]["additional_delay_minutes"], 5.0)
        self.assertEqual(c0["delay"]["total_estimated_duration_minutes"], 25.0)
        self.assertIn("advisory estimate", c0["delay"]["delay_source"].lower())
        self.assertIn("nominal advisory buffer", c0["delay"]["calculation_method"].lower())

        # Baseline duration is kept strictly separate
        self.assertEqual(c0["delay"]["baseline_duration_minutes"], 20.0)

    # --------------------------------------------------------------------------
    # 8. Multiple journeys with different alert outcomes
    # --------------------------------------------------------------------------
    def test_multiple_journeys_isolated_alert_outcomes(self):
        # 1. Flood incident on SV Road (Western Suburbs)
        live_events.append({
            "observation_id": "INC-WESTERN-FLOOD",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "ACTIVE",
            "traffic": {"road_passability": "CLOSED"}
        })

        # Journey 1: Borivali to Andheri (Intersects flood zone)
        payload_1 = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460},
            "user_journey_id": "WESTERN-CORRIDOR"
        }
        resp_1 = self.client.post("/api/routes/alerts/evaluate", json=payload_1)
        self.assertEqual(resp_1.status_code, 200)
        data_1 = resp_1.json()
        self.assertEqual(data_1["alert"]["alert_level"], "RED")
        self.assertTrue(data_1["alert"]["should_alert"])
        self.assertEqual(len(data_1["impacting_incidents"]), 1)

        # Journey 2: Powai to BKC (Eastern corridor; unaffected)
        self.mock_route_svc.plan_routes.return_value = [CLEAR_POWAI_BKC_ROUTE]
        self.mock_route_svc.plan_route.return_value = CLEAR_POWAI_BKC_ROUTE

        payload_2 = {
            "origin": {"latitude": 19.1176, "longitude": 72.9060},
            "destination": {"latitude": 19.0657, "longitude": 72.8687},
            "user_journey_id": "EASTERN-CORRIDOR"
        }
        resp_2 = self.client.post("/api/routes/alerts/evaluate", json=payload_2)
        self.assertEqual(resp_2.status_code, 200)
        data_2 = resp_2.json()
        self.assertEqual(data_2["alert"]["alert_level"], "GREEN")
        self.assertFalse(data_2["alert"]["should_alert"])
        self.assertEqual(len(data_2["impacting_incidents"]), 0)

    # --------------------------------------------------------------------------
    # 9. Malformed requests and coordinate validation errors
    # --------------------------------------------------------------------------
    def test_malformed_coordinates_and_api_errors(self):
        # Invalid latitude > 90
        bad_lat = {
            "origin": {"latitude": 195.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp_lat = self.client.post("/api/routes/evaluate", json=bad_lat)
        self.assertIn(resp_lat.status_code, [400, 422])
        self.assertTrue("latitude" in str(resp_lat.json()).lower())

        # Invalid longitude < -180
        bad_lon = {
            "origin": {"latitude": 19.2307, "longitude": -195.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp_lon = self.client.post("/api/routes/evaluate", json=bad_lon)
        self.assertIn(resp_lon.status_code, [400, 422])
        self.assertTrue("longitude" in str(resp_lon.json()).lower())

        # Missing origin field
        missing_origin = {
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp_missing = self.client.post("/api/routes/evaluate", json=missing_origin)
        self.assertEqual(resp_missing.status_code, 422)

    # --------------------------------------------------------------------------
    # 10. Dedicated POST /api/routes/alternatives/evaluate Endpoint
    # --------------------------------------------------------------------------
    def test_dedicated_alternatives_endpoint(self):
        # Confirmed closure on SV Road
        live_events.append({
            "observation_id": "INC-SV-CLOSURE-DET",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali West",
            "latitude": 19.2050,
            "longitude": 72.8420,
            "status": "ACTIVE",
            "traffic": {"road_passability": "CLOSED"}
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460},
            "candidate_radius_m": 500.0
        }
        resp = self.client.post("/api/routes/alternatives/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        alt = resp.json()
        self.assertEqual(alt["status"], "ALTERNATIVE_RECOMMENDED")
        self.assertTrue(alt["alternatives_available"])
        self.assertEqual(alt["recommended_candidate_id"], "candidate-1")
        self.assertEqual(alt["total_candidates"], 2)
        self.assertEqual(len(alt["candidates"]), 2)

        # Check provenance and disclaimers
        self.assertIn("OSRM alternatives", alt["limitations_and_disclaimer"])
        self.assertIn("free-flow", alt["limitations_and_disclaimer"])


if __name__ == "__main__":
    unittest.main()
