"""
Unit and Integration Tests for UrbanPulse OSRM Routing Foundation
================================================================
Verifies:
1. Coordinate validation (bounds [-90, 90], [-180, 180], finite values).
2. OSRM coordinate ordering (longitude,latitude; NOT latitude,longitude).
3. Environment-based configuration (OSRM_BASE_URL, OSRM_TIMEOUT_SECONDS, OSRM_PROFILE).
4. Response parsing: GeoJSON LineString, distance in m/km, duration in s/min, baseline duration disclaimer.
5. No-route response handling (OSRM NoRoute / NoSegment / empty routes).
6. Timeout handling (httpx.TimeoutException -> OSRMTimeoutError -> 504).
7. Service error handling (httpx.ConnectError / 5xx -> OSRMServiceError -> 502).
8. FastAPI POST /api/routes/plan endpoint contracts and status codes.
"""

import os
import sys
import math
import unittest
from unittest.mock import patch, MagicMock
import httpx
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from route_service import (
    OSRMConfig,
    OSRMRouteService,
    Coordinate,
    CoordinateValidationError,
    RouteNotFoundError,
    OSRMTimeoutError,
    OSRMServiceError,
    OSRMInvalidResponseError,
    RoutingError,
    validate_coordinate_pair,
    set_route_service
)
from api_service import app


# Sample OSRM Mock Data
SAMPLE_OSRM_SUCCESS_RESPONSE = {
    "code": "Ok",
    "routes": [
        {
            "geometry": {
                "coordinates": [
                    [72.8567, 19.2307],
                    [72.8580, 19.2250],
                    [72.8620, 19.1800],
                    [72.8697, 19.1136]
                ],
                "type": "LineString"
            },
            "legs": [
                {
                    "distance": 15420.2,
                    "duration": 1823.4,
                    "summary": "Western Express Highway",
                    "weight": 1823.4,
                    "steps": [
                        {
                            "distance": 850.2,
                            "duration": 120.5,
                            "geometry": {
                                "coordinates": [
                                    [72.8567, 19.2307],
                                    [72.8580, 19.2250]
                                ],
                                "type": "LineString"
                            },
                            "maneuver": {
                                "bearing_after": 170,
                                "bearing_before": 0,
                                "location": [72.8567, 19.2307],
                                "type": "depart"
                            },
                            "mode": "driving",
                            "name": "Link Road"
                        },
                        {
                            "distance": 14570.0,
                            "duration": 1702.9,
                            "geometry": {
                                "coordinates": [
                                    [72.8580, 19.2250],
                                    [72.8697, 19.1136]
                                ],
                                "type": "LineString"
                            },
                            "maneuver": {
                                "bearing_after": 165,
                                "bearing_before": 170,
                                "location": [72.8580, 19.2250],
                                "type": "turn"
                            },
                            "mode": "driving",
                            "name": "Western Express Highway"
                        }
                    ]
                }
            ],
            "weight_name": "routability",
            "weight": 1823.4,
            "duration": 1823.4,
            "distance": 15420.2
        }
    ],
    "waypoints": [
        {
            "distance": 4.5,
            "hint": "abc123hint",
            "location": [72.8567, 19.2307],
            "name": "Link Road"
        },
        {
            "distance": 2.1,
            "hint": "xyz789hint",
            "location": [72.8697, 19.1136],
            "name": "JVLR Junction"
        }
    ]
}

SAMPLE_OSRM_NO_ROUTE_RESPONSE = {
    "code": "NoRoute",
    "message": "Impossible route between points"
}

SAMPLE_OSRM_NO_SEGMENT_RESPONSE = {
    "code": "NoSegment",
    "message": "Could not find a matching segment for coordinate"
}


class TestCoordinateValidation(unittest.TestCase):
    """Verifies strict coordinate validation for latitudes and longitudes."""

    def test_valid_coordinates_pass(self):
        # Mumbai coordinates
        validate_coordinate_pair(19.2307, 72.8567, label="Origin")
        validate_coordinate_pair(19.1136, 72.8697, label="Destination")
        # Boundary extremes
        validate_coordinate_pair(-90.0, -180.0)
        validate_coordinate_pair(90.0, 180.0)
        validate_coordinate_pair(0.0, 0.0)

    def test_invalid_latitude_raises(self):
        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(90.1, 72.8567)

        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(-90.1, 72.8567)

    def test_invalid_longitude_raises(self):
        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(19.2307, 180.5)

        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(19.2307, -180.5)

    def test_non_finite_coordinates_raise(self):
        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(float("nan"), 72.8567)

        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(19.2307, float("inf"))

        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair(float("-inf"), 72.8567)

    def test_non_numeric_coordinates_raise(self):
        with self.assertRaises(CoordinateValidationError):
            validate_coordinate_pair("invalid", 72.8567)  # type: ignore


class TestCoordinateOrdering(unittest.TestCase):
    """Verifies that OSRM coordinates are strictly ordered longitude,latitude."""

    def setUp(self):
        self.service = OSRMRouteService(OSRMConfig(base_url="http://router.project-osrm.org"))

    def test_coordinate_ordering_in_url(self):
        origin_lat, origin_lon = 19.2307, 72.8567
        dest_lat, dest_lon = 19.1136, 72.8697

        url = self.service.build_osrm_url(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon
        )

        # OSRM requires: {origin_lon},{origin_lat};{dest_lon},{dest_lat}
        expected_coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        self.assertIn(expected_coords, url)

        # Must NOT be formatted as latitude,longitude
        reversed_coords = f"{origin_lat},{origin_lon};{dest_lat},{dest_lon}"
        self.assertNotIn(reversed_coords, url)

        # Must include full geometry and GeoJSON format
        self.assertIn("overview=full", url)
        self.assertIn("geometries=geojson", url)
        self.assertIn("steps=true", url)


class TestOSRMConfiguration(unittest.TestCase):
    """Verifies environment-based configuration for OSRM."""

    def test_default_configuration(self):
        # Clear env vars if set
        with patch.dict(os.environ, {}, clear=True):
            config = OSRMConfig()
            self.assertEqual(config.base_url, "http://router.project-osrm.org")
            self.assertEqual(config.timeout_seconds, 10.0)
            self.assertEqual(config.profile, "driving")

    def test_custom_environment_configuration(self):
        custom_env = {
            "OSRM_BASE_URL": "http://custom-osrm-host:5000/",
            "OSRM_TIMEOUT_SECONDS": "15.5",
            "OSRM_PROFILE": "car"
        }
        with patch.dict(os.environ, custom_env, clear=True):
            config = OSRMConfig()
            # Trailing slash should be stripped
            self.assertEqual(config.base_url, "http://custom-osrm-host:5000")
            self.assertEqual(config.timeout_seconds, 15.5)
            self.assertEqual(config.profile, "car")


class TestResponseParsing(unittest.TestCase):
    """Verifies that OSRM JSON response is correctly parsed into RoutePlanResponse."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.service = OSRMRouteService(client=self.mock_client)

    def test_successful_parsing(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSRM_SUCCESS_RESPONSE
        self.mock_client.get.return_value = mock_resp

        result = self.service.plan_route(
            origin_lat=19.2307,
            origin_lon=72.8567,
            dest_lat=19.1136,
            dest_lon=72.8697
        )

        # Verify origin and destination
        self.assertEqual(result.origin.latitude, 19.2307)
        self.assertEqual(result.origin.longitude, 72.8567)
        self.assertEqual(result.destination.latitude, 19.1136)
        self.assertEqual(result.destination.longitude, 72.8697)

        # Verify geometry is exact GeoJSON LineString (not fabricated)
        self.assertEqual(result.geometry.type, "LineString")
        self.assertEqual(
            result.geometry.coordinates,
            SAMPLE_OSRM_SUCCESS_RESPONSE["routes"][0]["geometry"]["coordinates"]
        )

        # Verify distance in metres and kilometres
        self.assertEqual(result.distance.meters, 15420.2)
        self.assertEqual(result.distance.kilometers, 15.42)

        # Verify baseline duration in seconds and minutes
        self.assertEqual(result.duration.seconds, 1823.4)
        self.assertEqual(result.duration.minutes, 30.39)
        self.assertFalse(result.duration.traffic_aware)
        self.assertIn("live traffic-aware ETA not included", result.duration.note)

        # Verify source metadata
        self.assertEqual(result.source.provider, "OSRM")
        self.assertEqual(result.source.profile, "driving")
        self.assertTrue(len(result.source.retrieval_timestamp) > 0)

        # Verify legs and steps
        self.assertEqual(len(result.legs), 1)
        leg = result.legs[0]
        self.assertEqual(leg.distance_meters, 15420.2)
        self.assertEqual(leg.duration_seconds, 1823.4)
        self.assertEqual(leg.summary, "Western Express Highway")
        self.assertIsNotNone(leg.steps)
        self.assertEqual(len(leg.steps), 2)
        self.assertEqual(leg.steps[0]["name"], "Link Road")


class TestNoRouteResponses(unittest.TestCase):
    """Verifies error handling when OSRM cannot find a route."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.service = OSRMRouteService(client=self.mock_client)

    def test_osrm_noroute_code_raises_routenotfound(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSRM_NO_ROUTE_RESPONSE
        self.mock_client.get.return_value = mock_resp

        with self.assertRaises(RouteNotFoundError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("No route found", str(ctx.exception))

    def test_osrm_nosegment_code_raises_routenotfound(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSRM_NO_SEGMENT_RESPONSE
        self.mock_client.get.return_value = mock_resp

        with self.assertRaises(RouteNotFoundError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("No route found", str(ctx.exception))

    def test_empty_routes_list_raises_routenotfound(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": "Ok", "routes": []}
        self.mock_client.get.return_value = mock_resp

        with self.assertRaises(RouteNotFoundError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("No route returned", str(ctx.exception))


class TestTimeoutsAndServiceErrors(unittest.TestCase):
    """Verifies timeouts, network failures, and 5xx errors."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.service = OSRMRouteService(client=self.mock_client)

    def test_timeout_raises_osrmtimeouterror(self):
        self.mock_client.get.side_effect = httpx.ReadTimeout("Request timed out")

        with self.assertRaises(OSRMTimeoutError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("timed out", str(ctx.exception).lower())

    def test_connect_error_raises_osrmserviceerror(self):
        self.mock_client.get.side_effect = httpx.ConnectError("Failed to connect")

        with self.assertRaises(OSRMServiceError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("unavailable or unreachable", str(ctx.exception))

    def test_500_server_error_raises_osrmserviceerror(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"message": "Internal OSRM error"}
        self.mock_client.get.return_value = mock_resp

        with self.assertRaises(OSRMServiceError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("server error", str(ctx.exception).lower())

    def test_malformed_non_json_raises_invalidresponse(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        self.mock_client.get.return_value = mock_resp

        with self.assertRaises(OSRMInvalidResponseError) as ctx:
            self.service.plan_route(19.2307, 72.8567, 19.1136, 72.8697)

        self.assertIn("Failed to parse OSRM response as JSON", str(ctx.exception))


class TestAPIEndpoint(unittest.TestCase):
    """End-to-end tests for FastAPI POST /api/routes/plan endpoint."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_service = OSRMRouteService(client=self.mock_client)
        set_route_service(self.mock_service)
        self.api_client = TestClient(app)

    def tearDown(self):
        set_route_service(None)

    def test_plan_route_endpoint_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSRM_SUCCESS_RESPONSE
        self.mock_client.get.return_value = mock_resp

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1136, "longitude": 72.8697}
        }
        response = self.api_client.post("/api/routes/plan", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["origin"]["latitude"], 19.2307)
        self.assertEqual(data["origin"]["longitude"], 72.8567)
        self.assertEqual(data["destination"]["latitude"], 19.1136)
        self.assertEqual(data["destination"]["longitude"], 72.8697)

        self.assertEqual(data["geometry"]["type"], "LineString")
        self.assertEqual(len(data["geometry"]["coordinates"]), 4)

        self.assertEqual(data["distance"]["meters"], 15420.2)
        self.assertEqual(data["distance"]["kilometers"], 15.42)

        self.assertEqual(data["duration"]["seconds"], 1823.4)
        self.assertEqual(data["duration"]["minutes"], 30.39)
        self.assertFalse(data["duration"]["traffic_aware"])

        self.assertEqual(data["source"]["provider"], "OSRM")
        self.assertEqual(len(data["legs"]), 1)
        self.assertEqual(len(data["legs"][0]["steps"]), 2)

    def test_plan_route_invalid_coordinate_validation(self):
        # Latitude out of bounds (> 90)
        payload = {
            "origin": {"latitude": 95.0, "longitude": 72.8567},
            "destination": {"latitude": 19.1136, "longitude": 72.8697}
        }
        response = self.api_client.post("/api/routes/plan", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_plan_route_no_route_returns_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSRM_NO_ROUTE_RESPONSE
        self.mock_client.get.return_value = mock_resp

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1136, "longitude": 72.8697}
        }
        response = self.api_client.post("/api/routes/plan", json=payload)
        self.assertEqual(response.status_code, 404)
        self.assertIn("No route found", response.json()["detail"])

    def test_plan_route_timeout_returns_504(self):
        self.mock_client.get.side_effect = httpx.ReadTimeout("Request timed out")

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1136, "longitude": 72.8697}
        }
        response = self.api_client.post("/api/routes/plan", json=payload)
        self.assertEqual(response.status_code, 504)
        self.assertIn("timed out", response.json()["detail"])

    def test_plan_route_unavailable_returns_502(self):
        self.mock_client.get.side_effect = httpx.ConnectError("Service unreachable")

        payload = {
            "origin": {"latitude": 19.2307, "longitude": 72.8567},
            "destination": {"latitude": 19.1136, "longitude": 72.8697}
        }
        response = self.api_client.post("/api/routes/plan", json=payload)
        self.assertEqual(response.status_code, 502)
        self.assertIn("unavailable or unreachable", response.json()["detail"])

    def test_root_endpoint_lists_route_plan(self):
        response = self.api_client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("plan_route", data["endpoints"])
        self.assertEqual(data["endpoints"]["plan_route"], "POST /api/routes/plan")


if __name__ == "__main__":
    unittest.main()
