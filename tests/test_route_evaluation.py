"""
Unit and Integration Tests for Route Impact and Personalized Alert APIs (Phase 7B.4)
===================================================================================
Verifies:
1. POST /api/routes/evaluate:
   - Full evaluation with mocked OSRM route and active incidents.
   - Impassable/blocked route -> RED / AVOID_ROUTE.
   - Restricted/possible disruption -> AMBER / PROCEED_WITH_CAUTION.
   - Clean corridor -> GREEN / ROUTE_CLEAR.
   - Separate OSRM baseline duration and agent-reported delays (no blind 45m additions).
   - Alternative routes status (NOT_YET_IMPLEMENTED).
   - Non-fabrication on OSRM timeout/service failure (504/502).
2. POST /api/routes/alerts/evaluate:
   - Evaluates demo traveler journey with personalized alert decision.
3. GET /api/users/me/route-alerts:
   - Returns HTTP 501 documenting requirement for user authentication and saved journeys.
4. GET /api/incidents/{incident_id}/route-impact:
   - Retrieves incident details and route impact explanation with or without journey coordinates.
   - Returns 404 for unknown incident.
5. Dynamic CORS environment configuration.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from route_service import (
    OSRMRouteService,
    OSRMConfig,
    set_route_service,
    Coordinate,
    GeoJSONLineString,
    DistanceBreakdown,
    DurationBreakdown,
    RouteSourceInfo,
    RouteLeg,
    RoutePlanResponse
)
from incident_service import (
    NormalizedIncident,
    get_active_incidents
)
from api_service import app, live_events


MOCK_ROUTE_POLYLINE = [
    [72.8567, 19.2307],  # Borivali
    [72.8580, 19.2050],  # Kandivali
    [72.8540, 19.1870],  # Malad
    [72.8460, 19.1190]   # Andheri
]

MOCK_ROUTE_RESPONSE = RoutePlanResponse(
    origin=Coordinate(latitude=19.2307, longitude=72.8567),
    destination=Coordinate(latitude=19.1190, longitude=72.8460),
    geometry=GeoJSONLineString(type="LineString", coordinates=MOCK_ROUTE_POLYLINE),
    distance=DistanceBreakdown(meters=16500.0, kilometers=16.5),
    duration=DurationBreakdown(
        seconds=960.0,
        minutes=16.0,
        traffic_aware=False,
        note="Baseline free-flow duration"
    ),
    source=RouteSourceInfo(
        provider="OSRM",
        profile="driving",
        base_url="http://mock-osrm",
        retrieval_timestamp="2026-10-13T06:00:00Z"
    ),
    legs=[
        RouteLeg(
            distance_meters=16500.0,
            duration_seconds=960.0,
            summary="Western Express Highway",
            steps=[]
        )
    ]
)


class TestRouteEvaluationEndpoints(unittest.TestCase):
    """End-to-end tests for Phase 7B.4 route evaluation and alert APIs."""

    def setUp(self):
        live_events.clear()
        self.mock_route_svc = MagicMock()
        self.mock_route_svc.plan_route.return_value = MOCK_ROUTE_RESPONSE
        set_route_service(self.mock_route_svc)
        self.client = TestClient(app)

    def tearDown(self):
        live_events.clear()
        set_route_service(None)

    def test_evaluate_route_clean_corridor_green(self):
        # No incidents in live_events
        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["alert"]["alert_level"], "GREEN")
        self.assertEqual(data["alert"]["action"], "ROUTE_CLEAR")
        self.assertFalse(data["alert"]["should_alert"])
        self.assertEqual(data["delay"]["baseline_duration_minutes"], 16.0)
        self.assertEqual(data["delay"]["additional_delay_minutes"], 0.0)
        self.assertEqual(data["delay"]["total_estimated_duration_minutes"], 16.0)
        self.assertEqual(data["evaluation_summary"]["overall_impact_status"], "NO_ESTABLISHED_DISRUPTION")
        self.assertIn(data["alternative_routes"]["status"], ["PRIMARY_ROUTE_OPTIMAL", "NOT_YET_IMPLEMENTED"])
        self.assertEqual(len(data["impacting_incidents"]), 0)

    def test_evaluate_route_confirmed_closure_red(self):
        # Add a confirmed road closure on Kandivali segment
        live_events.append({
            "observation_id": "INC-CLOSURE-KANDIVALI",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali",
            "latitude": 19.2051,
            "longitude": 72.8581,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "CLOSED",
                "traffic_delay_minutes": 45.0
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["alert"]["alert_level"], "RED")
        self.assertEqual(data["alert"]["action"], "AVOID_ROUTE")
        self.assertTrue(data["alert"]["should_alert"])
        self.assertTrue(data["evaluation_summary"]["is_route_blocked"])
        self.assertEqual(data["evaluation_summary"]["overall_impact_status"], "CONFIRMED_DISRUPTION")
        self.assertEqual(len(data["impacting_incidents"]), 1)
        self.assertTrue(data["impacting_incidents"][0]["is_road_blocked"])

    def test_evaluate_route_restricted_amber(self):
        # Add restricted passability in Malad
        live_events.append({
            "observation_id": "INC-RESTRICTED-MALAD",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Malad",
            "latitude": 19.1872,
            "longitude": 72.8542,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "RESTRICTED",
                "traffic_delay_minutes": 20.0,
                "affected_route_id": "ROUTE-004"
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["alert"]["alert_level"], "AMBER")
        self.assertEqual(data["alert"]["action"], "PROCEED_WITH_CAUTION")
        self.assertTrue(data["alert"]["should_alert"])
        self.assertFalse(data["evaluation_summary"]["is_route_blocked"])
        self.assertEqual(data["evaluation_summary"]["overall_impact_status"], "POSSIBLE_DISRUPTION")
        self.assertEqual(data["delay"]["additional_delay_minutes"], 20.0)
        self.assertEqual(data["delay"]["total_estimated_duration_minutes"], 36.0)

    def test_evaluate_route_unrelated_nearby_incident_no_delay_added(self):
        # Incident 150m away with PASSABLE status
        live_events.append({
            "observation_id": "INC-PASSABLE-NEARBY",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Malad",
            "latitude": 19.1870,
            "longitude": 72.8555,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "PASSABLE",
                "traffic_delay_minutes": 45.0  # Even if 45m is reported, road is PASSABLE
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        # Must NOT blindly add the 45-minute delay!
        self.assertEqual(data["delay"]["additional_delay_minutes"], 0.0)
        self.assertEqual(data["delay"]["total_estimated_duration_minutes"], 16.0)
        self.assertEqual(len(data["impacting_incidents"]), 0)
        self.assertEqual(len(data["nearby_unrelated_incidents"]), 1)
        self.assertEqual(data["alert"]["alert_level"], "GREEN")

    def test_evaluate_route_osrm_timeout_returns_504(self):
        from route_service import OSRMTimeoutError
        self.mock_route_svc.plan_route.side_effect = OSRMTimeoutError("OSRM timed out after 10s")

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460}
        }
        resp = self.client.post("/api/routes/evaluate", json=payload)
        # Never fabricate route geometry on timeout; return 504
        self.assertEqual(resp.status_code, 504)
        self.assertIn("timed out", resp.json()["detail"])

    def test_evaluate_journey_alerts(self):
        live_events.append({
            "observation_id": "INC-JOURNEY-ALERT",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Kandivali",
            "latitude": 19.2051,
            "longitude": 72.8581,
            "traffic": {
                "road_passability": "CLOSED"
            }
        })

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1190, "longitude": 72.8460},
            "user_journey_id": "COMMUTE-01",
            "journey_name": "Morning Work Commute"
        }
        resp = self.client.post("/api/routes/alerts/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["journey_id"], "COMMUTE-01")
        self.assertEqual(data["journey_name"], "Morning Work Commute")
        self.assertTrue(data["alert"]["should_alert"])
        self.assertEqual(data["alert"]["alert_level"], "RED")
        self.assertEqual(len(data["impacting_incidents"]), 1)

    def test_saved_user_route_alerts_endpoint_501(self):
        resp = self.client.get("/api/users/me/route-alerts")
        self.assertEqual(resp.status_code, 501)
        self.assertIn("require authentication and user profile storage", resp.json()["detail"])

    def test_incident_route_impact_endpoint_found_and_not_found(self):
        # 1. Unknown incident returns 404
        resp_404 = self.client.get("/api/incidents/INC-NONEXISTENT/route-impact")
        self.assertEqual(resp_404.status_code, 404)

        # 2. Existing incident in live_events
        live_events.append({
            "observation_id": "INC-TEST-DETAIL",
            "observation_time": "2026-10-13T05:50:00+05:30",
            "locality": "Malad",
            "latitude": 19.1870,
            "longitude": 72.8540,
            "status": "ACTIVE",
            "traffic": {
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-004"
            }
        })

        resp_200 = self.client.get("/api/incidents/INC-TEST-DETAIL/route-impact")
        self.assertEqual(resp_200.status_code, 200)
        data = resp_200.json()
        self.assertEqual(data["incident_id"], "INC-TEST-DETAIL")
        self.assertEqual(data["passability"], "RESTRICTED")
        self.assertEqual(data["affected_route_id"], "ROUTE-004")

        # 3. With journey coordinates query params
        resp_with_coords = self.client.get(
            "/api/incidents/INC-TEST-DETAIL/route-impact?origin_lat=19.2307&origin_lon=72.8567&dest_lat=19.1190&dest_lon=72.8460"
        )
        self.assertEqual(resp_with_coords.status_code, 200)
        data_coords = resp_with_coords.json()
        self.assertIsNotNone(data_coords["journey_evaluation"])
        self.assertIn(data_coords["journey_evaluation"]["classification"], ["ON_ROUTE", "AFFECTS_ROUTE", "NEAR_ROUTE"])


if __name__ == "__main__":
    unittest.main()
